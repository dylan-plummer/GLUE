r"""
Integration models
"""

import os
from pathlib import Path
from typing import Mapping, Optional

import dill
import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData

from ..data import estimate_balancing_weight
from ..typehint import Kws
from ..utils import config, logged
from .base import Model
from .dx import integration_consistency
from .nn import autodevice
from .scclue import SCCLUEModel
from .scglue import PairedSCGLUEModel, SCGLUEModel
from .plugins import EmbeddingVisualizer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@logged
def configure_dataset(
        adata: AnnData, prob_model: str,
        use_highly_variable: bool = True,
        use_layer: Optional[str] = None,
        use_rep: Optional[str] = None,
        use_batch: Optional[str] = None,
        use_depth: Optional[str] = None,
        use_cell_type: Optional[str] = None,
        use_dsc_weight: Optional[str] = None,
        use_obs_names: bool = False
) -> None:
    r"""
    Configure dataset for model training

    Parameters
    ----------
    adata
        Dataset to be configured
    prob_model
        Probabilistic generative model used by the decoder,
        must be one of ``{"Normal", "ZIN", "ZILN", "NB", "ZINB"}``.
    use_highly_variable
        Whether to use highly variable features
    use_layer
        Data layer to use (key in ``adata.layers``)
    use_rep
        Data representation to use as the first encoder transformation
        (key in ``adata.obsm``)
    use_batch
        Data batch to use (key in ``adata.obs``)
    use_cell_type
        Data cell type to use (key in ``adata.obs``)
    use_dsc_weight
        Discriminator sample weight to use (key in ``adata.obs``)
    use_obs_names
        Whether to use ``obs_names`` to mark paired cells across
        different datasets

    Note
    -----
    The ``use_rep`` option applies to encoder inputs, but not the decoders,
    which are always fitted on data in the original space.
    """
    if config.ANNDATA_KEY in adata.uns:
        configure_dataset.logger.warning(
            "`configure_dataset` has already been called. "
            "Previous configuration will be overwritten!"
        )
    data_config = {}
    data_config["prob_model"] = prob_model
    if use_highly_variable:
        if "highly_variable" not in adata.var:
            raise ValueError("Please mark highly variable features first!")
        data_config["use_highly_variable"] = True
        data_config["features"] = adata.var.query("highly_variable").index.to_numpy().tolist()
    else:
        data_config["use_highly_variable"] = False
        data_config["features"] = adata.var_names.to_numpy().tolist()
    if use_layer:
        if use_layer not in adata.layers:
            raise ValueError("Invalid `use_layer`!")
        data_config["use_layer"] = use_layer
    else:
        data_config["use_layer"] = None
    if use_rep:
        if use_rep not in adata.obsm:
            raise ValueError("Invalid `use_rep`!")
        data_config["use_rep"] = use_rep
        data_config["rep_dim"] = adata.obsm[use_rep].shape[1]
    else:
        data_config["use_rep"] = None
        data_config["rep_dim"] = None
    if use_batch:
        if use_batch not in adata.obs:
            raise ValueError("Invalid `use_batch`!")
        data_config["use_batch"] = use_batch
        data_config["batches"] = pd.Index(
            adata.obs[use_batch]
        ).dropna().drop_duplicates().sort_values().to_numpy()  # AnnData does not support saving pd.Index in uns
    else:
        data_config["use_batch"] = None
        data_config["batches"] = None
    if use_depth:
        if use_depth not in adata.obs:
            raise ValueError("Invalid `use_depth`!")
        data_config["use_depth"] = use_depth
    else:
        data_config["use_depth"] = None 

    if use_cell_type:
        if use_cell_type not in adata.obs:
            raise ValueError("Invalid `use_cell_type`!")
        data_config["use_cell_type"] = use_cell_type
        data_config["cell_types"] = pd.Index(
            adata.obs[use_cell_type]
        ).dropna().drop_duplicates().sort_values().to_numpy()  # AnnData does not support saving pd.Index in uns
    else:
        data_config["use_cell_type"] = None
        data_config["cell_types"] = None
    if use_dsc_weight:
        if use_dsc_weight not in adata.obs:
            raise ValueError("Invalid `use_dsc_weight`!")
        data_config["use_dsc_weight"] = use_dsc_weight
    else:
        data_config["use_dsc_weight"] = None
    data_config["use_obs_names"] = use_obs_names
    adata.uns[config.ANNDATA_KEY] = data_config


def load_model(fname: os.PathLike) -> Model:
    r"""
    Load model from file

    Parameters
    ----------
    fname
        Specifies path to the file

    Returns
    -------
    model
        Loaded model
    """
    fname = Path(fname)
    with fname.open("rb") as f:
        model = dill.load(f)
    model.upgrade()  # pylint: disable=no-member
    model.net.device = autodevice()  # pylint: disable=no-member
    return model


@logged
def fit_SCGLUE(
        adatas: Mapping[str, AnnData], graph: nx.Graph, model: type = SCGLUEModel,
        init_kws: Kws = None, compile_kws: Kws = None, fit_kws: Kws = None,
        balance_kws: Kws = None, log_wandb: bool = False
) -> SCGLUEModel:
    r"""
    Fit GLUE model to integrate single-cell multi-omics data

    Parameters
    ----------
    adatas
        Single-cell datasets (indexed by modality name)
    graph
        Guidance graph
    model
        Model class, must be one of
        {:class:`scglue.models.scglue.SCGLUEModel`,
        :class:`scglue.models.scglue.PairedSCGLUEModel`}
    init_kws
        Model initialization keyword arguments
        (see the constructor of the ``model`` class,
        either :class:`scglue.models.scglue.SCGLUEModel`
        or :class:`scglue.models.scglue.PairedSCGLUEModel`)
    compile_kws
        Model compile keyword arguments
        (see the ``compile`` method of the ``model`` class,
        either :meth:`scglue.models.scglue.SCGLUEModel.compile`
        or :meth:`scglue.models.scglue.PairedSCGLUEModel.compile`)
    fit_kws
        Model fitting keyword arguments
        (see :meth:`scglue.models.scglue.SCGLUEModel.fit`)
    balance_kws
        Balancing weight estimation keyword arguments
        (see :func:`scglue.data.estimate_balancing_weight`)

    Returns
    -------
    model
        Fitted model object
    """
    init_kws = init_kws or {}
    compile_kws = compile_kws or {}
    fit_kws = fit_kws or {}
    balance_kws = balance_kws or {}

    fit_SCGLUE.logger.info("Pretraining SCGLUE model...")
    pretrain_init_kws = init_kws.copy()
    pretrain_init_kws.update({"shared_batches": False})
    pretrain_fit_kws = fit_kws.copy()
    pretrain_fit_kws.update({"align_burnin": np.inf, "safe_burnin": False})
    if "directory" in pretrain_fit_kws:
        pretrain_fit_kws["directory"] = \
            os.path.join(pretrain_fit_kws["directory"], "pretrain")

    pretrain = model(adatas, sorted(graph.nodes), **pretrain_init_kws)
    n_params = count_parameters(pretrain.net)
    fit_SCGLUE.logger.info(f"Number of trainable parameters: {int(n_params / 1e6)}M")
    pretrain.compile(**compile_kws)
    if log_wandb:
        import wandb
        wandb.watch(pretrain.net, log_freq=100, log="all")
    type_dict = dict(graph.nodes(data='feature_type', default='n/a'))
    chrom_dict = dict(graph.nodes(data='chrom', default='n/a'))
    weight_dict = dict(graph.nodes(data='chrom_pos', default=1.0))
    node_attr_df = {'type': [], 'chrom': [], 'pos': [], 'degree': []}
    for node in sorted(graph.nodes):
        node_attr_df['type'].append(type_dict[node])
        node_attr_df['chrom'].append(chrom_dict[node])
        node_attr_df['pos'].append(weight_dict[node])
        node_attr_df['degree'].append(graph.degree[node])
    node_attr_df = pd.DataFrame(node_attr_df, index=sorted(graph.nodes))
    print(node_attr_df)
    # create anndata for node attributes using g2v.vrepr as init X
    node_attr_anndata = AnnData(pretrain.net.g2v.vrepr.cpu().detach().numpy(), obs=node_attr_df)
    print(node_attr_anndata)
    embedding_viz = EmbeddingVisualizer(adatas['rna'], pretrain.modalities['rna'],
                                        adatas['hic'], pretrain.modalities['hic'], 
                                        graph, sorted(graph.nodes),
                                        node_attr_anndata,
                                        latent_dim=init_kws['latent_dim'],
                                        prefix='pretrain',
                                        save_interval=fit_kws.get('save_interval', 10))
    pretrain.fit(adatas, graph, **pretrain_fit_kws, plugins=[embedding_viz])
    if "directory" in pretrain_fit_kws:
        pretrain.save(os.path.join(pretrain_fit_kws["directory"], "pretrain.dill"))

    fit_SCGLUE.logger.info("Estimating balancing weight...")
    for k, adata in adatas.items():
        adata.obsm[f"X_{config.TMP_PREFIX}"] = pretrain.encode_data(k, adata)
    if init_kws.get("shared_batches"):
        use_batch = set(
            adata.uns[config.ANNDATA_KEY]["use_batch"]
            for adata in adatas.values()
        )
        use_batch = use_batch.pop() if len(use_batch) == 1 else None
    else:
        use_batch = None
    estimate_balancing_weight(
        *adatas.values(), use_rep=f"X_{config.TMP_PREFIX}", use_batch=use_batch,
        key_added="balancing_weight", **balance_kws
    )
    for adata in adatas.values():
        adata.uns[config.ANNDATA_KEY]["use_dsc_weight"] = "balancing_weight"
        del adata.obsm[f"X_{config.TMP_PREFIX}"]

    fit_SCGLUE.logger.info("Fine-tuning SCGLUE model...")
    finetune_fit_kws = fit_kws.copy()
    if "directory" in finetune_fit_kws:
        finetune_fit_kws["directory"] = \
            os.path.join(finetune_fit_kws["directory"], "fine-tune")

    finetune = model(adatas, sorted(graph.nodes), **init_kws)
    finetune.adopt_pretrained_model(pretrain)
    finetune.compile(**compile_kws)
    if log_wandb:
        import wandb
        wandb.watch(pretrain.net, log_freq=100, log="all")
    fit_SCGLUE.logger.debug("Increasing random seed by 1 to prevent idential data order...")
    finetune.random_seed += 1
    node_attr_anndata = AnnData(pretrain.net.g2v.vrepr.cpu().detach().numpy(), obs=node_attr_df)
    print(node_attr_anndata)
    embedding_viz = EmbeddingVisualizer(adatas['rna'], pretrain.modalities['rna'],
                                        adatas['hic'], pretrain.modalities['hic'],
                                        graph, sorted(graph.nodes), 
                                        node_attr_anndata,
                                        latent_dim=init_kws['latent_dim'],
                                        prefix='finetune',
                                        save_interval=fit_kws.get('save_interval', 10))
    finetune.fit(adatas, graph, **finetune_fit_kws, plugins=[embedding_viz])
    if "directory" in finetune_fit_kws:
        finetune.save(os.path.join(finetune_fit_kws["directory"], "fine-tune.dill"))

    return finetune
