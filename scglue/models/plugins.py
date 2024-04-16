r"""
Training plugins
"""

import os
import math
import pathlib
import shutil
from typing import Iterable, Optional

import wandb
import ignite
import ignite.contrib.handlers.tensorboard_logger as tb
import parse
import torch
import numpy as np
import pandas as pd
import networkx as nx
import anndata as ad
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..utils import config, logged
from ..num import normalize_edges
from .base import Trainer, TrainingPlugin
from .data import AnnDataset, DataLoader, GraphDataset
from ..data import transfer_labels

EPOCH_COMPLETED = ignite.engine.Events.EPOCH_COMPLETED
TERMINATE = ignite.engine.Events.TERMINATE
COMPLETED = ignite.engine.Events.COMPLETED


def get_closest_peak_to_gene(gene_name, rna, peaks):
    try:
        loc = rna.var.loc[gene_name]
    except KeyError:
        print('Could not find loci', gene_name)
        return None
    chrom = loc["chrom"]
    chromStart = loc["chromStart"]
    peaks['in_chr'] = peaks['chrom'] == chrom
    peaks['dist'] = peaks['chromStart'].apply(lambda s: abs(s - chromStart))
    peaks.loc[~peaks['in_chr'], 'dist'] = 1e9  # set distance to 1e9 if not in same chromosome
    return peaks['dist'].argmin()


class Tensorboard(TrainingPlugin):

    r"""
    Training logging via tensorboard
    """

    def attach(
            self, net: torch.nn.Module, trainer: Trainer,
            train_engine: ignite.engine.Engine,
            val_engine: ignite.engine.Engine,
            train_loader: Iterable,
            val_loader: Optional[Iterable],
            directory: pathlib.Path
    ) -> None:
        tb_directory = directory / "tensorboard"
        if tb_directory.exists():
            shutil.rmtree(tb_directory)

        tb_logger = tb.TensorboardLogger(
            log_dir=tb_directory,
            flush_secs=config.TENSORBOARD_FLUSH_SECS
        )
        tb_logger.attach(
            train_engine,
            log_handler=tb.OutputHandler(
                tag="train", metric_names=trainer.required_losses
            ), event_name=EPOCH_COMPLETED
        )
        if val_engine:
            tb_logger.attach(
                val_engine,
                log_handler=tb.OutputHandler(
                    tag="val", metric_names=trainer.required_losses
                ), event_name=EPOCH_COMPLETED
            )
        train_engine.add_event_handler(COMPLETED, tb_logger.close)


@logged
class EarlyStopping(TrainingPlugin):

    r"""
    Early stop model training when loss no longer decreases

    Parameters
    ----------
    monitor
        Loss to monitor
    patience
        Patience to stop early
    burnin
        Burn-in epochs to skip before initializing early stopping
    wait_n_lrs
        Wait n learning rate scheduling events before starting early stopping
    """

    def __init__(
            self, monitor: str, patience: int,
            burnin: int = 0, wait_n_lrs: int = 0
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.burnin = burnin
        self.wait_n_lrs = wait_n_lrs

    def attach(
            self, net: torch.nn.Module, trainer: Trainer,
            train_engine: ignite.engine.Engine,
            val_engine: ignite.engine.Engine,
            train_loader: Iterable,
            val_loader: Optional[Iterable],
            directory: pathlib.Path
    ) -> None:
        for item in directory.glob("checkpoint_*.pt"):
            item.unlink()

        score_engine = val_engine if val_engine else train_engine
        score_function = lambda engine: -score_engine.state.metrics[self.monitor]
        event_filter = (
            lambda engine, event: event > self.burnin and engine.state.n_lrs >= self.wait_n_lrs
        ) if self.wait_n_lrs else (
            lambda engine, event: event > self.burnin
        )
        event = EPOCH_COMPLETED(event_filter=event_filter)  # pylint: disable=not-callable
        train_engine.add_event_handler(
            event, ignite.handlers.Checkpoint(
                {"net": net, "trainer": trainer},
                ignite.handlers.DiskSaver(
                    directory, atomic=True, create_dir=True, require_empty=False
                ), score_function=score_function,
                filename_pattern="checkpoint_{global_step}.pt",
                n_saved=config.CHECKPOINT_SAVE_NUMBERS,
                global_step_transform=ignite.handlers.global_step_from_engine(train_engine)
            )
        )
        train_engine.add_event_handler(
            event, ignite.handlers.EarlyStopping(
                patience=self.patience,
                score_function=score_function,
                trainer=train_engine
            )
        )

        @train_engine.on(COMPLETED | TERMINATE)
        def _(engine):
            nan_flag = any(
                not bool(torch.isfinite(item).all())
                for item in (engine.state.output or {}).values()
            )
            ckpts = sorted([
                parse.parse("checkpoint_{epoch:d}.pt", item.name).named["epoch"]
                for item in directory.glob("checkpoint_*.pt")
            ], reverse=True)
            if ckpts and nan_flag and train_engine.state.epoch == ckpts[0]:
                self.logger.warning(
                    "The most recent checkpoint \"%d\" can be corrupted by NaNs, "
                    "will thus be discarded.", ckpts[0]
                )
                ckpts = ckpts[1:]
            if ckpts:
                self.logger.info("Restoring checkpoint \"%d\"...", ckpts[0])
                loaded = torch.load(directory / f"checkpoint_{ckpts[0]}.pt")
                net.load_state_dict(loaded["net"])
                trainer.load_state_dict(loaded["trainer"])
            else:
                self.logger.info(
                    "No usable checkpoint found. "
                    "Skipping checkpoint restoration."
                )


@logged
class LRScheduler(TrainingPlugin):

    r"""
    Reduce learning rate on loss plateau

    Parameters
    ----------
    *optims
        Optimizers
    monitor
        Loss to monitor
    patience
        Patience to reduce learning rate
    burnin
        Burn-in epochs to skip before initializing learning rate scheduling
    """

    def __init__(
            self, *optims: torch.optim.Optimizer, monitor: str = None,
            patience: int = None, burnin: int = 0
    ) -> None:
        super().__init__()
        if monitor is None:
            raise ValueError("`monitor` must be specified!")
        self.monitor = monitor
        if patience is None:
            raise ValueError("`patience` must be specified!")
        self.schedulers = [
            ReduceLROnPlateau(optim, patience=patience, verbose=True)
            for optim in optims
        ]
        self.burnin = burnin

    def attach(
            self, net: torch.nn.Module, trainer: Trainer,
            train_engine: ignite.engine.Engine,
            val_engine: ignite.engine.Engine,
            train_loader: Iterable,
            val_loader: Optional[Iterable],
            directory: pathlib.Path
    ) -> None:
        score_engine = val_engine if val_engine else train_engine
        event_filter = lambda engine, event: event > self.burnin
        for scheduler in self.schedulers:
            scheduler.last_epoch = self.burnin
        train_engine.state.n_lrs = 0

        @train_engine.on(EPOCH_COMPLETED(event_filter=event_filter))  # pylint: disable=not-callable
        def _():
            update_flags = set()
            for scheduler in self.schedulers:
                old_lr = scheduler.optimizer.param_groups[0]["lr"]
                scheduler.step(score_engine.state.metrics[self.monitor])
                new_lr = scheduler.optimizer.param_groups[0]["lr"]
                update_flags.add(new_lr != old_lr)
            if len(update_flags) != 1:
                raise RuntimeError("Learning rates are out of sync!")
            if update_flags.pop():
                train_engine.state.n_lrs += 1
                self.logger.info("Learning rate reduction: step %d", train_engine.state.n_lrs)

# add plugin for visualizing embeddings after each epoch
@logged
class EmbeddingVisualizer(TrainingPlugin):
    def __init__(
            self, 
            rna, rna_data_config,
            hic, hic_data_config,
            graph, vertices,
            features_ad: ad.AnnData,
            latent_dim=64,
            prefix='pretrain',
            out_dir: str = 'tmp_imgs'
    ) -> None:
        super().__init__()
        self.rna = rna
        self.rna_data_config = rna_data_config
        self.hic = hic
        self.hic.obs['sorted_celltype'] = self.hic.obs['celltype']
        self.graph = graph
        self.vertices = pd.Index(vertices)
        self.features_ad = features_ad
        self.alpha_beta_val = False
        if self.hic.obs_names.str.startswith('alpha_').any() or self.hic.obs_names.str.startswith('beta_').any():
            sorted_alpha_mask = self.hic.obs_names.str.startswith('alpha_')
            sorted_beta_mask = self.hic.obs_names.str.startswith('beta_')
            # convert back to str
            self.hic.obs['sorted_celltype'] = self.hic.obs['sorted_celltype'].astype(str)
            self.hic.obs.loc[sorted_alpha_mask, 'sorted_celltype'] = 'alpha (sorted)'
            self.hic.obs.loc[sorted_beta_mask, 'sorted_celltype'] = 'beta (sorted)'
            self.alpha_beta_val = True
        
        self.data_config = hic_data_config
        # self.mha = torch.nn.MultiheadAttention(
        #     latent_dim, num_heads=1, bias=False
        # )
        
        self.prefix = prefix
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def attach(
            self, net: torch.nn.Module, trainer: Trainer,
            train_engine: ignite.engine.Engine,
            val_engine: ignite.engine.Engine,
            train_loader: Iterable,
            val_loader: Optional[Iterable],
            directory: pathlib.Path
    ) -> None:
        train_engine.state.n_lrs = 0

        @train_engine.on(EPOCH_COMPLETED)  # pylint: disable=not-callable
        def _():
            epoch = train_engine.state.epoch
            save_interval = 10
            if epoch % save_interval == 0:
                save_i = int(epoch // save_interval)  # for animation
                # get hic embeddings
                net.eval()
                encoder = net.x2u['hic']
                data = AnnDataset([self.hic], [self.data_config], mode='eval', getitem_size=128)
                data_loader = DataLoader(
                    data, batch_size=1, shuffle=False,
                    num_workers=config.DATALOADER_NUM_WORKERS,
                    pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
                    persistent_workers=False
                )
                result = []
                for x, xrep, *_ in data_loader:
                    u = encoder(
                        x.to(net.device, non_blocking=True),
                        xrep.to(net.device, non_blocking=True),
                        lazy_normalizer=True
                    )[0]
                    result.append(u.mean.detach().cpu())
                result = torch.cat(result).numpy()
                self.hic.obsm['X_glue'] = result

                # get rna embeddings
                encoder = net.x2u['rna']
                data = AnnDataset([self.rna], [self.rna_data_config], mode='eval', getitem_size=128)
                data_loader = DataLoader(
                    data, batch_size=1, shuffle=False,
                    num_workers=config.DATALOADER_NUM_WORKERS,
                    pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
                    persistent_workers=False
                )
                result = []
                for x, xrep, *_ in data_loader:
                    u = encoder(
                        x.to(net.device, non_blocking=True),
                        xrep.to(net.device, non_blocking=True),
                        lazy_normalizer=True
                    )[0]
                    result.append(u.mean.detach().cpu())
                result = torch.cat(result).numpy()
                self.rna.obsm['X_glue'] = result

                # get feature embeddings
                print('Embedding features...')
                graph = GraphDataset(self.graph, self.vertices)
                enorm = torch.as_tensor(
                    normalize_edges(graph.eidx, graph.ewt),
                    device=net.device
                )
                esgn = torch.as_tensor(graph.esgn, device=net.device)
                eidx = torch.as_tensor(graph.eidx, device=net.device)

                v = net.g2v(eidx, enorm, esgn)
                self.features_ad.X = v.mean.detach().cpu().numpy()

                net.train()

                # visualize some params
                try:  # in case we are running an experiment without key conv params
                    params_out_dir = f'{self.out_dir}/params'
                    os.makedirs(params_out_dir, exist_ok=True)
                    decoder = net.u2x['hic']
                    key_conv_weights = []
                    for layer in decoder.key_convs:
                        key_conv_weights.append(layer.weight.detach().cpu().numpy())
                    key_conv_weights = np.hstack(key_conv_weights) # shape (embedding size, n_strata, 3)
                    # plot n_strata heatmap of key conv weights
                    n_plots = min(10, key_conv_weights.shape[1] - 1)
                    fig, axs = plt.subplots(1, n_plots, figsize=(n_plots, 3))
                    vmin = np.min(key_conv_weights)
                    vmax = np.max(key_conv_weights)
                    flattened_mat = key_conv_weights[:, 1:, :].reshape(key_conv_weights.shape[0], -1)
                    # compute order based on dendrogram
                    linkage = sns.clustermap(flattened_mat, row_cluster=True, col_cluster=False, figsize=(10, 10), method='average', metric='cosine', cbar=False)
                    plt.close()
                    order = linkage.dendrogram_row.reordered_ind
                    for i in range(n_plots):
                        ax = axs[i]
                        sns.heatmap(key_conv_weights[order, i + 1, :], ax=ax, cmap='coolwarm', center=0, vmin=vmin, vmax=vmax, cbar=i == n_plots - 1)
                        # remove ticks labels and grid
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.grid(False)

                    plt.tight_layout()
                    fig.savefig(f'{params_out_dir}/{self.prefix}_key_conv_weights_{save_i}.png')
                    plt.close()
                except:
                    pass


                # visualize feature embeddings
                feature_out_dir = f'{self.out_dir}/features'
                os.makedirs(feature_out_dir, exist_ok=True)
                x_pca = PCA(n_components=2).fit_transform(self.features_ad.X)
                self.features_ad.obsm['X_pca'] = x_pca

                sc.set_figure_params()
                fig = sc.pl.pca(self.features_ad, color=["chrom", "type", "pos", "degree"], ncols=2, wspace=0.5, return_fig=True)
                fig.savefig(f'{feature_out_dir}/{self.prefix}_pca_features_{save_i}.png')
                plt.close()

                # keep_chroms = ['chr19', 'chr20', 'chr21']

                # chr20_ad = self.features_ad[self.features_ad.obs['chrom'].isin(keep_chroms)]
                # sc.pp.neighbors(chr20_ad)
                # sc.tl.umap(chr20_ad)
                # fig = sc.pl.umap(chr20_ad, color=["chrom", "type", "pos", "degree"], ncols=2, wspace=0.5, return_fig=True)
                # fig.savefig(f'{feature_out_dir}/{self.prefix}_umap_features_{save_i}.png')
                # plt.close()


                # plot hic feature matrix
                mat_out_dir = f'{self.out_dir}/features/hic_feature_matrix'
                os.makedirs(mat_out_dir, exist_ok=True)
                hic_mat = self.features_ad[self.features_ad.obs['type'] == 'Hi-C', :]
                fig, ax = plt.subplots()
                sns.heatmap(hic_mat.X[:128, :], cmap='Spectral', center=0, ax=ax)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylabel('Hi-C features')
                ax.set_xlabel('Feature embeddings')
                # remove grid 
                ax.grid(False)
                plt.savefig(f'{mat_out_dir}/{self.prefix}_hic_feature_matrix_{save_i}.png')
                plt.close()

                # plot a few neighborhood examples
                # example_genes = ['GAD1', 'INS', 'GCG', 'CEL', 'LOXL4', 'WFS1', 'PPY', 'EPSTI1', 'PPARG', 'ANK1', 'TSPAN8', 'LPP',
                #                  'OTUD3', 'LRRTM3', 'BAACL', 'SLC14A1', 'ARNTL2', 'IRX2']
                example_genes = ['NR2F1', 'CNTN4', 'SYT1', 'GAD1']
                graph_out_dir = f'{self.out_dir}/loop_graph'
                for g in example_genes:
                    if g not in self.vertices:
                        continue
                    gene_out_dir = f'{graph_out_dir}/{g}'
                    gene_graph_dir = f'{gene_out_dir}/graph'
                    gene_out_dir_pca = f'{gene_out_dir}/pca'
                    gene_out_dir_umap = f'{gene_out_dir}/umap'
                    os.makedirs(gene_out_dir_pca, exist_ok=True)
                    os.makedirs(gene_out_dir_umap, exist_ok=True)
                    os.makedirs(gene_graph_dir, exist_ok=True)
                    gene_neighborhood = [g]
                    bin_idx = get_closest_peak_to_gene(g, self.rna, self.hic.var)
                    n_strata = len(net.u2x['hic'].key_convs)
                    for k in range(n_strata):
                        gene_neighborhood.append(self.hic.var.iloc[bin_idx + k].name)
                        gene_neighborhood.append(self.hic.var.iloc[bin_idx - k].name)
                    for node in self.graph.neighbors(g):
                        if node in self.vertices:
                            gene_neighborhood.append(node)
                            for neighbor in self.graph.neighbors(node):
                                if neighbor in self.vertices:
                                    gene_neighborhood.append(neighbor)
                                    for anchor in self.graph.neighbors(neighbor):
                                        if anchor in self.vertices:
                                            gene_neighborhood.append(anchor)
                    
                    gene_neighborhood = list(set(gene_neighborhood))
                    neighborhood_ad = self.features_ad[self.features_ad.obs.index.isin(gene_neighborhood)].copy()
                    # visualize graph with edges
                    gene_graph = nx.Graph(self.graph.subgraph(gene_neighborhood))
                    # remove self loops
                    gene_graph.remove_edges_from(nx.selfloop_edges(gene_graph))
                    fig, ax = plt.subplots()
                    graph_pos = nx.spring_layout(gene_graph)
                    nx.draw_networkx_nodes(gene_graph, pos=graph_pos, nodelist=neighborhood_ad.obs[neighborhood_ad.obs['type'] == 'RNA'].index, node_color='lightgreen', node_size=100)
                    nx.draw_networkx_nodes(gene_graph, pos=graph_pos, nodelist=neighborhood_ad.obs[neighborhood_ad.obs['type'] == 'Hi-C'].index, node_color='red', node_size=50)
                    nx.draw_networkx_edges(gene_graph, pos=graph_pos, edge_color='gray')
                    nx.draw_networkx_labels(gene_graph, pos=graph_pos, font_size=8, font_color='black')
                    # turn off ax grid
                    ax.grid(False)
                    plt.savefig(f'{gene_graph_dir}/{self.prefix}_graph_{g}_{save_i}.png')
                    plt.close()
                    # first visualize pca and umap of only bins and genes
                    x_pca = PCA(n_components=2).fit_transform(neighborhood_ad.X)
                    neighborhood_ad.obsm['X_pca'] = x_pca
                    #sc.set_figure_params()
                    fig = sc.pl.pca(neighborhood_ad, color=["chrom", "type", "pos", "degree"], ncols=2, wspace=0.5, return_fig=True, color_map='jet')
                    fig.savefig(f'{gene_out_dir_pca}/{self.prefix}_pca_features_{save_i}.png')
                    plt.close()
                    try:
                        sc.pp.neighbors(neighborhood_ad, metric="cosine")
                        sc.tl.umap(neighborhood_ad)
                        fig = sc.pl.umap(neighborhood_ad, color=["chrom", "type", "pos", "degree"], ncols=2, wspace=0.5, return_fig=True, color_map='jet')
                        fig.savefig(f'{gene_out_dir_umap}/{self.prefix}_umap_features_{save_i}.png')
                        plt.close()
                    except Exception as e:
                        print(e)
                        pass
                    # now transform features at each strata
                    per_strata_out_dir_pca = f'{gene_out_dir}/pca_strata'
                    per_strata_out_dir_umap = f'{gene_out_dir}/umap_strata'
                    per_strata_out_dir_graph = f'{gene_out_dir}/graph_strata'
                    os.makedirs(per_strata_out_dir_pca, exist_ok=True)
                    os.makedirs(per_strata_out_dir_umap, exist_ok=True)
                    os.makedirs(per_strata_out_dir_graph, exist_ok=True)
                    n_strata = len(net.u2x['hic'].key_convs)
                    neighborhood_ad.obs['strata'] = 0
                    per_strata_ads = [neighborhood_ad]
                    for k in range(n_strata):
                        # get only hi-c features
                        strata_ad = self.features_ad[self.features_ad.obs['type'] == 'Hi-C', :].copy()
                        strata_ad_mask = strata_ad.obs.index.isin(neighborhood_ad.obs.index)
                        v = strata_ad.X
                        v = torch.as_tensor(v, device=net.device)
                        key_conv = net.u2x['hic'].key_convs[k]
                        feats = key_conv(v.t()).t()
                        strata_ad.X = feats.detach().cpu().numpy()
                        strata_ad.obs['strata'] = k + 1
                        strata_ad.obs['type'] = 'Hi-C_strata'
                        per_strata_ads.append(strata_ad[strata_ad_mask, :])
                    per_strata_ad = ad.concat(per_strata_ads)
                    x_pca = PCA(n_components=2).fit_transform(per_strata_ad.X)
                    per_strata_ad.obsm['X_pca'] = x_pca
                    #sc.set_figure_params()
                    fig = sc.pl.pca(per_strata_ad, color=["type", "pos", "degree", "strata"], ncols=2, wspace=0.5, return_fig=True, color_map='jet')
                    fig.savefig(f'{per_strata_out_dir_pca}/{self.prefix}_pca_features_{save_i}.png')
                    plt.close()

                    ax = sc.pl.pca(per_strata_ad, color=["type"], show=False)
                    texts = []
                    for node in gene_neighborhood:
                        try:
                            node_type = per_strata_ad.obs.loc[node, 'type']
                            # if node is a gene, draw its name
                            if node_type == 'RNA':
                                x_loc = per_strata_ad.obsm['X_pca'][per_strata_ad.obs.index == node, 0]
                                y_loc = per_strata_ad.obsm['X_pca'][per_strata_ad.obs.index == node, 1]
                                texts.append(ax.text(x_loc, y_loc, node, fontsize=10))
                        except Exception as e:
                            pass
                    # Label selected genes on the plot
                    _ = adjust_text(
                        texts,
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1),
                        ax=ax,
                    )
                    plt.savefig(f'{per_strata_out_dir_pca}/{self.prefix}_labeled_pca_features_{save_i}.png')
                    plt.close()
                    # try:
                    #     sc.pp.neighbors(per_strata_ad)
                    #     # sc.tl.draw_graph(per_strata_ad, layout='kk')
                    #     # fig = sc.pl.draw_graph(per_strata_ad, color=["type", "pos", "degree", "strata"], ncols=2, wspace=0.5, return_fig=True, color_map='jet', edges=True, )
                    #     # fig.savefig(f'{per_strata_out_dir_graph}/{self.prefix}_graph_features_{save_i}.png')
                    #     # plt.close()

                    #     sc.tl.umap(per_strata_ad)
                    #     fig = sc.pl.umap(per_strata_ad, color=["type", "pos", "degree", "strata"], ncols=2, wspace=0.5, return_fig=True, color_map='jet')
                    #     fig.savefig(f'{per_strata_out_dir_umap}/{self.prefix}_umap_features_{save_i}.png')
                    #     plt.close()
                    # except Exception as e:
                    #     print(e)
                    #     pass



                #now plot the transformed features at each strata
                # n_strata = len(net.u2x['hic'].key_convs)
                # per_strata_ads = []
                
                # for k in range(n_strata):
                #     # get only hi-c features
                #     strata_ad = self.features_ad[self.features_ad.obs['type'] == 'Hi-C', :].copy()
                #     v = strata_ad.X
                #     v = torch.as_tensor(v, device=net.device)
                #     key_conv = net.u2x['hic'].key_convs[k]
                #     #qk = net.u2x['hic'].key_layers[k](v)
                #     #v = self.strata_attn(qk, qk, v)
                #     #v = net.u2x['hic'].strata_attn(qk, qk, v)
                #     feats = key_conv(v.t()).t()
                #     strata_ad.X = feats.detach().cpu().numpy()
                #     strata_ad.obs['strata'] = k
                #     chr20_ad = strata_ad[strata_ad.obs['chrom'].isin(keep_chroms)]
                #     per_strata_ads.append(chr20_ad)
                #     # x_pca = PCA(n_components=2).fit_transform(strata_ad.X)
                #     # strata_ad.obsm['X_pca'] = x_pca

                #     # sc.set_figure_params()
                #     # fig = sc.pl.pca(strata_ad, color=["chrom", "type", "pos", "degree"], ncols=2, wspace=0.5, return_fig=True)
                #     # fig.savefig(f'{out_dir}/{self.prefix}_pca_features_{save_i}.png')
                #     # plt.close()

                #     # # plot hic feature matrix
                #     # mat_out_dir = f'{self.out_dir}/features/hic_feature_matrix_strata_{k}'
                #     # os.makedirs(mat_out_dir, exist_ok=True)
                #     # fig, ax = plt.subplots()
                #     # sns.heatmap(strata_ad.X[:128, :], cmap='Spectral', center=0, ax=ax)
                #     # ax.set_xticks([])
                #     # ax.set_yticks([])
                #     # # remove grid 
                #     # ax.grid(False)
                #     # ax.set_ylabel('Hi-C features')
                #     # ax.set_xlabel('Feature embeddings')
                #     # plt.savefig(f'{mat_out_dir}/{self.prefix}_hic_feature_matrix_{save_i}.png')
                #     # plt.close()
                # chr20_ad = ad.concat(per_strata_ads)
                # print(chr20_ad)
                # # compute PCA
                # x_pca = PCA(n_components=2).fit_transform(chr20_ad.X)
                # chr20_ad.obsm['X_pca'] = x_pca

                # fig = sc.pl.pca(chr20_ad, color=["chrom", "type", "pos", "strata"], ncols=2, wspace=0.5, return_fig=True)
                # fig.savefig(f'{feature_out_dir}/{self.prefix}_umap_strata_features_{save_i}.png')
                # plt.close()

                # visualize attention matrices
                # attn_size = 64
                # n_rows = 6
                # attn_out_dir = f'{self.out_dir}/attn'
                # os.makedirs(attn_out_dir, exist_ok=True)
                # v = self.features_ad[self.features_ad.obs['type'] == 'Hi-C', :].X
                # v = torch.as_tensor(v, device=net.device)
                # fig, axs = plt.subplots(n_rows, len(net.u2x['hic'].key_layers), figsize=(len(net.u2x['hic'].key_layers) * 2, n_rows * 2))
                # for row_i in range(n_rows):
                #     #feats_in = torch.as_tensor(v[row_i * attn_size:(row_i + 1) * attn_size, :], device=net.device)
                #     mean_mats = []
                #     for key_layer, key_conv in zip(net.u2x['hic'].key_layers, net.u2x['hic'].key_convs):
                #         #feats_in = key_conv(v.t()).t()
                #         qk = key_layer(v)[row_i * attn_size:(row_i + 1) * attn_size, :]
                #         attn_mat = torch.matmul(qk, qk.T) / math.sqrt(qk.shape[1])
                #         #attn_mat = attn_mat.softmax(dim=1)
                #         #attn_mat = attn_mat.softmax(dim=1).detach().cpu().numpy()
                #         #_, attn_mat = self.mha(qk, qk, feats_in)
                #         attn_mat = attn_mat.detach().cpu().numpy()
                #         #mean_attn = attn_mat.mean(axis=0)
                #         mean_mats.append(attn_mat)
                #     try:
                #         vmax = np.max(mean_mats)
                        
                #         for i, ax in enumerate(axs[row_i]):
                #             mean_attn = mean_mats[i]
                #             sns.heatmap(mean_attn, ax=ax, cmap='inferno', square=True, cbar=False, vmin=0, vmax=vmax)
                #             # turn off ax grid
                #             ax.grid(False)
                #             ax.set_xticks([])
                #             ax.set_yticks([])
                #             if row_i == 0:
                #                 ax.set_title(f'strata {i + 1}')
                #     except Exception as e:
                #         print(e)
                #         print(mean_mats)
                #         pass
                # fig.savefig(f'{attn_out_dir}/{self.prefix}_attn_{save_i}.png')
                # plt.close()
                
                # if epoch > 1:
                #     chr10_11_ad = self.features_ad[self.features_ad.obs['chrom'].isin(['chr10', 'chr11'])]
                #     sc.pp.neighbors(chr10_11_ad, metric="cosine")
                #     sc.tl.umap(chr10_11_ad)
                #     fig = sc.pl.umap(chr10_11_ad, color=["chrom", "type", "pos", "degree"], ncols=2, wspace=0.5, return_fig=True)
                #     fig.savefig(f'{feature_out_dir}/{self.prefix}_umap_features_{save_i}.png')
                #     plt.close()

                transfer_labels(self.rna, self.hic, "celltype", use_rep="X_glue", n_neighbors=10, key_added="pred_celltype")

                # compute pca
                pca_out_dir = f'{self.out_dir}/pca'
                os.makedirs(pca_out_dir, exist_ok=True)
                x_pca = PCA(n_components=2).fit_transform(self.hic.obsm['X_glue'])
                self.hic.obsm['X_pca'] = x_pca

                # cluster embeddings
                sc.tl.leiden(self.hic)
                #self.hic.obs['kmeans'] = KMeans(n_clusters=len(self.hic.obs['celltype'].unique())).fit_predict(self.hic.obsm['X_glue'])
                self.hic.obs['agglomerative'] = AgglomerativeClustering(n_clusters=len(self.hic.obs['celltype'].unique())).fit_predict(self.hic.obsm['X_glue'])
                # set leiden to int to avoid clash with colormap
                self.hic.obs['leiden'] = self.hic.obs['leiden'].astype(int)

                wspace = 0.45
                palette = None
                use_pfc_palette = True
                pfc_palette = {
                    "L2/3": [230, 25, 75],
                    "L4": [60, 180, 75],
                    "L5": [255, 225, 25],
                    "L6": [0, 130, 200],
                    "Ndnf": [245, 130, 49],
                    "Vip": [145, 30, 180],
                    "Pvalb": [70, 240, 240],
                    "Sst": [240, 50, 230]}
                pfc_palette = {k: [v / 255.0 for v in pfc_palette[k]] for k in pfc_palette}
                pfc_palette['Neu'] = 'gray'
                pfc_palette['Neu_rna'] = 'gray'
                for celltype in self.hic.obs['celltype'].unique():
                    if celltype not in pfc_palette:
                        use_pfc_palette = False
                        break
                if use_pfc_palette:
                    palette = pfc_palette
                elif len(self.hic.obs['celltype'].unique()) > 10:
                    wspace = 1.2
                    palette = sc.pl.palettes.godsnot_102

                sc.set_figure_params()
                fig = sc.pl.pca(self.hic, color=["sorted_celltype", "pred_celltype", "leiden", "agglomerative", "depth", "balancing_weight" if "balancing_weight" in self.hic.obs.columns else "batch"], palette=palette, ncols=3, wspace=wspace, return_fig=True)
                fig.savefig(f'{pca_out_dir}/{self.prefix}_pca_{save_i}.png')
                plt.close()

                umap_out_dir = f'{self.out_dir}/umap'
                os.makedirs(umap_out_dir, exist_ok=True)
                sc.pp.neighbors(self.hic, use_rep="X_glue", metric="cosine")
                sc.tl.umap(self.hic)
                
                fig = sc.pl.umap(self.hic, color=["sorted_celltype", "pred_celltype", "leiden", "agglomerative", "depth", "balancing_weight" if "balancing_weight" in self.hic.obs.columns else "batch"], palette=palette, ncols=3, wspace=wspace, return_fig=True)
                fig.savefig(f'{umap_out_dir}/{self.prefix}_umap_{save_i}.png')
                plt.close()

                
                # map celltypes to integers
                celltypes = list(sorted(self.hic.obs['celltype'].unique())) + ['Other']
                rna_celltypes = list(sorted(self.rna.obs['celltype'].unique())) + ['Other']
                celltype_map = {c: i for i, c in enumerate(celltypes)}
                rna_celltype_map = {c: i for i, c in enumerate(rna_celltypes)}
                self.hic.obs['celltype_int'] = self.hic.obs['celltype'].map(celltype_map)
                self.hic.obs['pred_celltype'] = self.hic.obs['pred_celltype'].apply(lambda s: s if s in celltypes else 'Other')
                self.hic.obs['pred_celltype_int'] = self.hic.obs['pred_celltype'].map(celltype_map)

                confusion_matrix_out_dir = f'{self.out_dir}/confusion_matrix'
                os.makedirs(confusion_matrix_out_dir, exist_ok=True)
                cmtx = confusion_matrix(self.hic.obs['celltype_int'], self.hic.obs['pred_celltype_int'], labels=list(range(len(celltypes))))
                fig, ax = plt.subplots(figsize=(15, 15))
                sns.heatmap(cmtx, annot=True, fmt=',d', xticklabels=celltypes, yticklabels=celltypes, square=True, ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                # turn off ax grid
                ax.grid(False)
                ax.xaxis.tick_top()
                fig.savefig(f'{confusion_matrix_out_dir}/{self.prefix}_confusion_{save_i}.png')
                plt.close()

                # measure ari
                ari_leiden = adjusted_rand_score(self.hic.obs['celltype_int'], self.hic.obs['leiden'])
                ari_kmeans = adjusted_rand_score(self.hic.obs['celltype_int'], self.hic.obs['agglomerative'])

                # measure ari in joint embedding space (usually better than in Hi-C only space)
                ari_joint = adjusted_rand_score(self.hic.obs['celltype_int'], self.hic.obs['pred_celltype_int'])
                f1_micro = f1_score(self.hic.obs['celltype_int'], self.hic.obs['pred_celltype_int'], average='micro', labels=list(range(len(celltypes))))
                f1_macro = f1_score(self.hic.obs['celltype_int'], self.hic.obs['pred_celltype_int'], average='macro', labels=list(range(len(celltypes))))
                accuracy = accuracy_score(self.hic.obs['celltype_int'], self.hic.obs['pred_celltype_int'])
                balanced_accuracy = balanced_accuracy_score(self.hic.obs['celltype_int'], self.hic.obs['pred_celltype_int'])

                val_accuracy = 0
                val_balanced_accuracy = 0
                val_ari = 0
                if self.alpha_beta_val:
                    sorted_hic = self.hic[self.hic.obs_names.str.startswith('alpha_') | self.hic.obs_names.str.startswith('beta_')]
                    sorted_rna = self.rna[self.rna.obs['celltype'].isin(['Alpha', 'Beta'])]
                    sorted_hic.obs['sorted_celltype'] = sorted_hic.obs['celltype']
                    celltypes = sorted(sorted_hic.obs['celltype'].unique())
                    celltype_map = {c: i for i, c in enumerate(celltypes)}
                    sorted_hic.obs['celltype_int'] = sorted_hic.obs['celltype'].map(celltype_map)
                    transfer_labels(sorted_rna, sorted_hic, "celltype", use_rep="X_glue", n_neighbors=5, key_added="pred_celltype_sorted")
                    sorted_hic.obs['pred_celltype_int'] = sorted_hic.obs['pred_celltype_sorted'].map(celltype_map)
                    # measure accuracy
                    val_accuracy = accuracy_score(sorted_hic.obs['celltype_int'], sorted_hic.obs['pred_celltype_int'])
                    val_balanced_accuracy = balanced_accuracy_score(sorted_hic.obs['celltype_int'], sorted_hic.obs['pred_celltype_int'])
                    val_ari = adjusted_rand_score(sorted_hic.obs['celltype_int'], sorted_hic.obs['pred_celltype_int'])
                    self.logger.info(f"Val Accuracy (joint): {val_accuracy:.2f}, Val Balanced Accuracy (joint): {val_balanced_accuracy:.2f}, Val ARI (joint): {val_ari:.2f}")
                    # replot
                    sorted_pca_out_dir = f'{self.out_dir}/pca_sorted'
                    os.makedirs(sorted_pca_out_dir, exist_ok=True)
                    sc.set_figure_params()
                    fig = sc.pl.pca(sorted_hic, color=["celltype", "pred_celltype", "depth", "balancing_weight" if "balancing_weight" in sorted_hic.obs.columns else "batch"], ncols=2, wspace=0.45, return_fig=True)
                    fig.savefig(f'{sorted_pca_out_dir}/{self.prefix}_pca_sorted_{save_i}.png')
                    plt.close()

                    sorted_umap_out_dir = f'{self.out_dir}/umap_sorted'
                    os.makedirs(sorted_umap_out_dir, exist_ok=True)
                    fig = sc.pl.umap(sorted_hic, color=["celltype", "pred_celltype", "depth", "balancing_weight" if "balancing_weight" in sorted_hic.obs.columns else "batch"], ncols=2, wspace=0.45, return_fig=True)
                    fig.savefig(f'{sorted_umap_out_dir}/{self.prefix}_umap_sorted_{save_i}.png')
                    plt.close()

                    # create sorted confusion matrix
                    sorted_cmtx = confusion_matrix(sorted_hic.obs['celltype_int'], sorted_hic.obs['pred_celltype_int'], labels=list(range(len(celltypes))))
                    sorted_confusion_matrix_out_dir = f'{self.out_dir}/confusion_matrix_sorted'
                    os.makedirs(sorted_confusion_matrix_out_dir, exist_ok=True)
                    fig, ax = plt.subplots(figsize=(15, 15))
                    sns.heatmap(sorted_cmtx, annot=True, fmt=',d', xticklabels=celltypes, yticklabels=celltypes, square=True, ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    # turn off ax grid
                    ax.grid(False)
                    ax.xaxis.tick_top()
                    fig.savefig(f'{sorted_confusion_matrix_out_dir}/{self.prefix}_confusion_sorted_{save_i}.png')
                    plt.close()
                
                self.logger.info(f"ARI (Hi-C only): {ari_kmeans:.2f}, ARI (joint): {ari_joint:.2f}, Accuracy (joint): {accuracy:.2f}, Balanced Accuracy (joint): {balanced_accuracy:.2f}")
                wandb.log({f"ari_{self.prefix}": ari_kmeans, 
                           f"ari_leiden_{self.prefix}": ari_leiden,
                           f"ari_joint_{self.prefix}": ari_joint, 
                            f"f1_micro_{self.prefix}": f1_micro,
                            f"f1_macro_{self.prefix}": f1_macro,
                           f"accuracy_{self.prefix}": accuracy,
                            f"balanced_accuracy_{self.prefix}": balanced_accuracy,
                            f"val_accuracy_{self.prefix}": val_accuracy,
                            f"val_balanced_accuracy_{self.prefix}": val_balanced_accuracy,
                            f"val_ari_{self.prefix}": val_ari,
                           "epoch": epoch}, step=epoch)
                

            

            
