r"""
GLUE component modules for single-cell omics data
"""

import collections
from abc import abstractmethod
from typing import Optional, Tuple

import random
import math
import torch
import torch.distributions as D
import torch.nn.functional as F

from ..num import EPS
from . import glue
from .nn import GraphConv, GraphAttent
from .prob import ZILN, ZIN, ZINB
from local_attention import LocalAttention


#-------------------------- Network modules for GLUE ---------------------------

class GraphEncoderWithNodeAttributes(glue.GraphEncoder):

    r"""
    Graph encoder with node attributes

    Parameters
    ----------
    vnum
        Number of vertices
    out_features
        Output dimensionality
    node_attr_dim
        Dimensionality of node attributes
    """

    def __init__(
            self, vnum: int, out_features: int, node_attr_dim: int
    ) -> None:
        super().__init__()
        self.vrepr = torch.nn.Parameter(torch.zeros(vnum, out_features))
        self.node_attr_proj = torch.nn.Linear(node_attr_dim, out_features)
        self.conv = GraphConv()
        self.loc = torch.nn.Linear(out_features, out_features)
        self.std_lin = torch.nn.Linear(out_features, out_features)

    def forward(
            self, eidx: torch.Tensor, enorm: torch.Tensor, esgn: torch.Tensor, node_attrs: torch.Tensor
    ) -> D.Normal:
        # Project node attributes to the same embedding space as vrepr
        node_attr_embeddings = self.node_attr_proj(node_attrs)
        # Combine node attributes with vrepr
        combined_vrepr = self.vrepr + node_attr_embeddings

        ptr = self.conv(combined_vrepr, eidx, enorm, esgn)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std)

class GraphEncoder(glue.GraphEncoder):

    r"""
    Graph encoder

    Parameters
    ----------
    vnum
        Number of vertices
    out_features
        Output dimensionality
    """

    def __init__(
            self, vnum: int, out_features: int
    ) -> None:
        super().__init__()
        self.vrepr = torch.nn.Parameter(torch.zeros(vnum, out_features))
        self.conv = GraphConv()
        self.conv2 = GraphConv()
        # self.conv = GraphAttent(out_features, out_features)
        # self.conv2 = GraphAttent(out_features, out_features)
        self.dropout = torch.nn.AlphaDropout(p=0.1)
        #self.ln = torch.nn.LayerNorm(out_features * 2)
        self.loc = torch.nn.Linear(out_features, out_features)
        self.std_lin = torch.nn.Linear(out_features, out_features)

    def forward(
            self, eidx: torch.Tensor, enorm: torch.Tensor, esgn: torch.Tensor
    ) -> D.Normal:
        ptr = self.conv(self.vrepr, eidx, enorm, esgn)
        ptr = F.selu(ptr)
        #ptr = self.dropout(ptr)
        ptr = self.conv2(ptr, eidx, enorm, esgn)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std)
    
class GraphEncoderMultiStrata(glue.GraphEncoder):

    r"""
    Graph encoder

    Parameters
    ----------
    vnum
        Number of vertices
    out_features
        Output dimensionality
    """

    def __init__(
            self, vnum: int, out_features: int, n_strata: int = 5
    ) -> None:
        super().__init__()
        per_strata_out_features = int(out_features / n_strata)
        self.vrepr = torch.nn.Parameter(torch.zeros(vnum, per_strata_out_features))
        self.conv = GraphConv()
        self.loc = torch.nn.Linear(per_strata_out_features, per_strata_out_features)
        self.std_lin = torch.nn.Linear(per_strata_out_features, per_strata_out_features)
        self.strata_convs = []
        self.strata_convs2 = []
        self.strata_vrepr = []
        self.strata_loc = []
        self.strata_std_lin = []
        for i in range(1, n_strata):
            self.strata_convs.append(GraphConv())
            self.strata_convs2.append(GraphConv())
            self.strata_loc.append(torch.nn.Linear(per_strata_out_features, per_strata_out_features))
            self.strata_std_lin.append(torch.nn.Linear(per_strata_out_features, per_strata_out_features))
        # so the params get registered
        self.strata_convs = torch.nn.ModuleList(self.strata_convs)
        self.strata_convs2 = torch.nn.ModuleList(self.strata_convs2)
        self.strata_loc = torch.nn.ModuleList(self.strata_loc)
        self.strata_std_lin = torch.nn.ModuleList(self.strata_std_lin)
        self.dropout = torch.nn.AlphaDropout(p=0.1)


    def forward(
            self, eidx: torch.Tensor, enorm: torch.Tensor, esgn: torch.Tensor
    ) -> D.Normal:
        ptr = self.conv(self.vrepr, eidx, enorm, esgn)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        locs = [loc]
        stds = [std]
        for i in range(len(self.strata_loc)):
            ptr = self.strata_convs[i](self.vrepr, eidx, enorm, esgn)
            ptr = F.selu(ptr)
            ptr = self.dropout(ptr)
            ptr = self.strata_convs2[i](ptr, eidx, enorm, esgn)
            strata_loc = self.strata_loc[i](ptr)
            strata_std = F.softplus(self.strata_std_lin[i](ptr)) + EPS
            locs.append(strata_loc)
            stds.append(strata_std)
        return D.Normal(torch.concat(locs, dim=1), torch.concat(stds, dim=1))


class GraphDecoder(glue.GraphDecoder):

    r"""
    Graph decoder
    """

    def forward(
            self, v: torch.Tensor, eidx: torch.Tensor, esgn: torch.Tensor
    ) -> D.Bernoulli:
        sidx, tidx = eidx  # Source index and target index
        logits = esgn * (v[sidx] * v[tidx]).sum(dim=1)
        return D.Bernoulli(logits=logits)


class DataEncoder(glue.DataEncoder):

    r"""
    Abstract data encoder

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def __init__(
            self, in_features: int, out_features: int,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2,
            downsample_min: float = 0.0,
            downsample_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.h_depth = h_depth
        ptr_dim = in_features
        for layer in range(self.h_depth):
            setattr(self, f"linear_{layer}", torch.nn.Linear(ptr_dim, h_dim))
            setattr(self, f"act_{layer}", torch.nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn_{layer}", torch.nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout_{layer}", torch.nn.Dropout(p=dropout))
            ptr_dim = h_dim
        self.loc = torch.nn.Linear(ptr_dim, out_features)
        self.std_lin = torch.nn.Linear(ptr_dim, out_features)
        self.downsample_min = downsample_min
        self.downsample_max = downsample_max
        self.downsample_prob = 0.5

    @abstractmethod
    def compute_l(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        r"""
        Compute normalizer

        Parameters
        ----------
        x
            Input data

        Returns
        -------
        l
            Normalizer
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def normalize(
            self, x: torch.Tensor, l: Optional[torch.Tensor]
    ) -> torch.Tensor:
        r"""
        Normalize data

        Parameters
        ----------
        x
            Input data
        l
            Normalizer

        Returns
        -------
        xnorm
            Normalized data
        """
        raise NotImplementedError  # pragma: no cover

    def forward(  # pylint: disable=arguments-differ
            self, x: torch.Tensor, xrep: torch.Tensor,
            lazy_normalizer: bool = True
    ) -> Tuple[D.Normal, Optional[torch.Tensor]]:
        r"""
        Encode data to sample latent distribution

        Parameters
        ----------
        x
            Input data
        xrep
            Alternative input data
        lazy_normalizer
            Whether to skip computing `x` normalizer (just return None)
            if `xrep` is non-empty

        Returns
        -------
        u
            Sample latent distribution
        normalizer
            Data normalizer

        Note
        ----
        Normalization is always computed on `x`.
        If xrep is empty, the normalized `x` will be used as input
        to the encoder neural network, otherwise xrep is used instead.
        """
        # if self.training:
        #     # only downsample during training
        #     if self.downsample_min > 0.0 and self.downsample_max < 1.0:
        #         # only downsample downsample_prob of the time
        #         if random.random() < self.downsample_prob:
        #             # # sample dropout prob from downsample min and max
        #             # p = random.uniform(self.downsample_min, self.downsample_max)
        #             # # Calculate probabilities for each batch by normalizing the counts
        #             # probabilities = x.float() / x.sum(dim=1, keepdim=True)

        #             # # Calculate the number of samples to take for each batch
        #             # num_samples = (x.sum(dim=1) * p).long()

        #             # # Sample from the distribution for each batch
        #             # samples = [torch.multinomial(prob, n, replacement=True) for prob, n in zip(probabilities, num_samples)]

        #             # # Create a new tensor with sampled counts for each batch
        #             # sampled_counts = torch.zeros_like(x)
        #             # for i, s in enumerate(samples):
        #             #     sampled_counts[i].scatter_add_(0, s, torch.ones_like(s, dtype=torch.float))
        #             # x = sampled_counts
        #             # perform dropout with probability inversely proportional to total counts
        #             p = random.uniform(self.downsample_min, self.downsample_max)
        #             probabilities = x.float() / (x.max(dim=1, keepdim=True)[0] * p)
        #             # clip probabilities to 1.0
        #             probabilities = torch.where(probabilities > 1.0, torch.ones_like(probabilities), probabilities)
        #             drop_mask = 1 - torch.bernoulli(probabilities)
        #             x = x * drop_mask
        if xrep.numel():
            l = None if lazy_normalizer else self.compute_l(x)
            ptr = xrep
        else:
            l = self.compute_l(x)
            ptr = self.normalize(x, l)
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std), l


class VanillaDataEncoder(DataEncoder):

    r"""
    Vanilla data encoder

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def compute_l(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        return None

    def normalize(
            self, x: torch.Tensor, l: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return x
    

class HiCDataEncoder(DataEncoder):

    r"""
    Data encoder for Hi-C data

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    TOTAL_COUNT = 5e4

    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)

    def normalize(
            self, x: torch.Tensor, l: torch.Tensor
    ) -> torch.Tensor:
        return (x * (self.TOTAL_COUNT / l)).log1p()
        #return x.log1p()
    
class HiCGraphConv(torch.nn.Module):

    r"""
    Graph convolution (propagation only)
    """

    def forward(
            self, input: torch.Tensor, eidx: torch.Tensor
            #enorm: torch.Tensor, esgn: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Forward propagation

        Parameters
        ----------
        input
            Input data (:math:`n_{vertices} \times n_{features}`)
        eidx
            Vertex indices of edges (:math:`2 \times n_{edges}`)
        enorm
            Normalized weight of edges (:math:`n_{edges}`)
        esgn
            Sign of edges (:math:`n_{edges}`)

        Returns
        -------
        result
            Graph convolution result (:math:`n_{vertices} \times n_{features}`)
        """
        sidx, tidx = eidx  # source index and target index
        message = input[:, sidx]  # batch * n_edges * n_features
        res = torch.zeros_like(input)
        # repeat target_idxs along batch dimension
        tidx = tidx.unsqueeze(0).expand_as(message)  # batch * n_edges * n_features
        res.scatter_add_(1, tidx, message)
        return res
    

class HiCDataEncoderGraphConv(glue.DataEncoder):

    r"""
    Data encoder for Hi-C data

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    TOTAL_COUNT = 5e4

    def __init__(self, in_features: int, out_features: int, h_depth: int = 2, h_dim: int = 256, dropout: float = 0.2, downsample_min: float = 0, downsample_max: float = 1,
                 strata_masks: list = []) -> None:
        super().__init__()
        self.h_depth = h_depth
        ptr_dim = h_dim * len(strata_masks)
        for layer in range(self.h_depth):
            setattr(self, f"linear_{layer}", torch.nn.Linear(ptr_dim, h_dim))
            setattr(self, f"act_{layer}", torch.nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn_{layer}", torch.nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout_{layer}", torch.nn.Dropout(p=dropout))
            ptr_dim = h_dim
        self.loc = torch.nn.Linear(ptr_dim, out_features)
        self.std_lin = torch.nn.Linear(ptr_dim, out_features)
        self.conv = HiCGraphConv()
        self.strata_masks = strata_masks
        #self.strata_idxs = torch.nn.ParameterList([torch.nn.Parameter(torch.as_tensor(s), requires_grad=False) for s in strata_masks])
        self.layer_norms = []
        self.strata_projections = []
        for strata_k in range(len(strata_masks)):
            #self.strata_idxs.append(torch.as_tensor(strata_masks[strata_k], device='cuda'))
            self.layer_norms.append(torch.nn.LayerNorm(h_dim))
            self.strata_projections.append(torch.nn.Linear(len(strata_masks[strata_k]), h_dim))
        self.layer_norms = torch.nn.ModuleList(self.layer_norms)
        self.strata_projections = torch.nn.ModuleList(self.strata_projections)
        self.max_strata_size = 0
        for strata in strata_masks:
            self.max_strata_size = max(self.max_strata_size, len(strata))


    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)

    def normalize(
            self, x: torch.Tensor, l: torch.Tensor
    ) -> torch.Tensor:
        return (x * (self.TOTAL_COUNT / l)).log1p()
    
    def forward(  # pylint: disable=arguments-differ
            self, x: torch.Tensor, xrep: torch.Tensor,
            lazy_normalizer: bool = True
    ) -> Tuple[D.Normal, Optional[torch.Tensor]]:
        if xrep.numel():
            l = None if lazy_normalizer else self.compute_l(x)
            ptr = xrep
        else:
            l = self.compute_l(x)
            ptr = self.normalize(x, l)
         
        conv_out = []
        for strata_k in range(len(self.strata_masks)):
            x = ptr[:, self.strata_masks[strata_k]]
            # for target_strata_k in range(len(self.strata_masks)):
            #     if target_strata_k == strata_k:  # skip self-loops
            #         continue
            #     if len(self.strata_masks[target_strata_k]) < 1:
            #         continue
            #     source_idxs = self.strata_idxs[target_strata_k]  # source indices of all strata_k bins
            #     target_idxs = torch.clip(source_idxs + target_strata_k - strata_k, 0, ptr.shape[1] - 1)  # targets are the same as sources, but shifted by the number of bins
            #     eidx = torch.stack([source_idxs, target_idxs], dim=0)
            #     x += self.conv(ptr, eidx)[:, self.strata_masks[strata_k]]
            # x /= len(self.strata_masks) - 1
            x = self.strata_projections[strata_k](x)
            x = self.layer_norms[strata_k](x)
            conv_out.append(x)
        ptr = torch.concat(conv_out, dim=1)
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std), l
    
    
    

class HiCDataEncoderStratified(glue.DataEncoder):

    r"""
    Data encoder for Hi-C data

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    TOTAL_COUNT = 5e4

    def __init__(self, in_features: int, out_features: int, h_depth: int = 2, h_dim: int = 256, dropout: float = 0.2, downsample_min: float = 0, downsample_max: float = 1,
                 strata_masks: list = []) -> None:
        super().__init__()
        self.h_depth = h_depth
        # self.loc_layers = []
        # self.std_layers = []
        #self.strata_conv_layers = []
        #for strata_k in range(len(strata_masks)):
            # ptr_dim = len(strata_masks[strata_k])
            # for layer in range(self.h_depth):
            #     setattr(self, f"linear_{layer}_{strata_k}", torch.nn.Linear(ptr_dim, h_dim))
            #     setattr(self, f"act_{layer}_{strata_k}", torch.nn.LeakyReLU(negative_slope=0.2))
            #     setattr(self, f"bn_{layer}_{strata_k}", torch.nn.BatchNorm1d(h_dim))
            #     setattr(self, f"dropout_{layer}_{strata_k}", torch.nn.Dropout(p=dropout))
            #     ptr_dim = h_dim
            # loc = torch.nn.Linear(ptr_dim, int(out_features / len(strata_masks)))
            # std_lin = torch.nn.Linear(ptr_dim, int(out_features / len(strata_masks)))
            # self.loc_layers.append(loc)
            # self.std_layers.append(std_lin)
            #strata_conv = torch.nn.Conv1d(1, 1, kernel_size=3, padding='same', bias=False)
            #self.strata_conv_layers.append(strata_conv)
        # self.loc_layers = torch.nn.ModuleList(self.loc_layers)
        # self.std_layers = torch.nn.ModuleList(self.std_layers)
        #self.strata_conv_layers = torch.nn.ModuleList(self.strata_conv_layers)
        ptr_dim = in_features
        for layer in range(self.h_depth):
            setattr(self, f"linear_{layer}", torch.nn.Linear(ptr_dim, h_dim))
            setattr(self, f"act_{layer}", torch.nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn_{layer}", torch.nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout_{layer}", torch.nn.Dropout(p=dropout))
            ptr_dim = h_dim
        self.loc = torch.nn.Linear(ptr_dim, out_features)
        self.std_lin = torch.nn.Linear(ptr_dim, out_features)
        # self.loc_strata_projection = torch.nn.Linear(len(strata_masks) * out_features, out_features)
        # self.std_strata_projection = torch.nn.Linear(len(strata_masks) * out_features, out_features)
        self.strata_masks = strata_masks
        self.max_strata_size = 0
        for strata in strata_masks:
            self.max_strata_size = max(self.max_strata_size, len(strata))


    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)

    def normalize(
            self, x: torch.Tensor, l: torch.Tensor
    ) -> torch.Tensor:
        return (x * (self.TOTAL_COUNT / l)).log1p()
        #return x.log1p()
    
    # def forward(  # pylint: disable=arguments-differ
    #         self, x: torch.Tensor, xrep: torch.Tensor,
    #         lazy_normalizer: bool = True
    # ) -> Tuple[D.Normal, Optional[torch.Tensor]]:
    #     if xrep.numel():
    #         l = None if lazy_normalizer else self.compute_l(x)
    #         ptr = xrep
    #     else:
    #         l = self.compute_l(x)
    #         ptr = self.normalize(x, l)
    #     locs = []
    #     stds = []
    #     for strata_k in range(len(self.strata_masks)):
    #         x = ptr[:, self.strata_masks[strata_k]]
    #         for layer in range(self.h_depth):
    #             x = getattr(self, f"linear_{layer}_{strata_k}")(x)
    #             x = getattr(self, f"act_{layer}_{strata_k}")(x)
    #             x = getattr(self, f"bn_{layer}_{strata_k}")(x)
    #             x = getattr(self, f"dropout_{layer}_{strata_k}")(x)
    
    #         loc = self.loc_layers[strata_k](x)
    #         std = self.std_layers[strata_k](x)
    #         locs.append(loc)
    #         stds.append(std)
    #     loc = torch.concat(locs, dim=1)
    #     std = torch.concat(stds, dim=1)
    #     # loc_full = self.loc_strata_projection(loc)
    #     # std_full = F.softplus(self.std_strata_projection(std)) + EPS
    #     std = F.softplus(std) + EPS
    #     return D.Normal(loc, std), l

    # def forward(  # pylint: disable=arguments-differ
    #         self, x: torch.Tensor, xrep: torch.Tensor,
    #         lazy_normalizer: bool = True
    # ) -> Tuple[D.Normal, Optional[torch.Tensor]]:
    #     if xrep.numel():
    #         l = None if lazy_normalizer else self.compute_l(x)
    #         ptr = xrep
    #     else:
    #         l = self.compute_l(x)
    #         ptr = self.normalize(x, l)
    #     locs = []
    #     stds = []
    #     stacked = []
    #     for strata_k in range(len(self.strata_masks)):
    #         x = ptr[:, self.strata_masks[strata_k]]
    #         x = x.unsqueeze(1)
    #         x = self.strata_conv_layers[strata_k](x)
    #         x = x.squeeze(1)
    #         stacked.append(x)
    #     ptr = torch.concat(stacked, dim=1)
    #     for layer in range(self.h_depth):
    #         ptr = getattr(self, f"linear_{layer}")(ptr)
    #         ptr = getattr(self, f"act_{layer}")(ptr)
    #         ptr = getattr(self, f"bn_{layer}")(ptr)
    #         ptr = getattr(self, f"dropout_{layer}")(ptr)
    #     loc = self.loc(ptr)
    #     std = F.softplus(self.std_lin(ptr)) + EPS
    #     return D.Normal(loc, std), l


class NBDataEncoder(DataEncoder):

    r"""
    Data encoder for negative binomial data

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    TOTAL_COUNT = 1e4

    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)

    def normalize(
            self, x: torch.Tensor, l: torch.Tensor
    ) -> torch.Tensor:
        return (x * (self.TOTAL_COUNT / l)).log1p()


class DataDecoder(glue.DataDecoder):

    r"""
    Abstract data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:  # pylint: disable=unused-argument
        super().__init__()

    @abstractmethod
    def forward(  # pylint: disable=arguments-differ
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Normal:
        r"""
        Decode data from sample and feature latent

        Parameters
        ----------
        u
            Sample latent
        v
            Feature latent
        b
            Batch index
        l
            Optional normalizer

        Returns
        -------
        recon
            Data reconstruction distribution
        """
        raise NotImplementedError  # pragma: no cover


class NormalDataDecoder(DataDecoder):

    r"""
    Normal data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.std_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Normal:
        scale = F.softplus(self.scale_lin[b])
        loc = scale * (u @ v.t()) + self.bias[b]
        std = F.softplus(self.std_lin[b]) + EPS
        return D.Normal(loc, std)


class ZINDataDecoder(NormalDataDecoder):

    r"""
    Zero-inflated normal data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> ZIN:
        scale = F.softplus(self.scale_lin[b])
        loc = scale * (u @ v.t()) + self.bias[b]
        std = F.softplus(self.std_lin[b]) + EPS
        return ZIN(self.zi_logits[b].expand_as(loc), loc, std)


class ZILNDataDecoder(DataDecoder):

    r"""
    Zero-inflated log-normal data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.std_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> ZILN:
        scale = F.softplus(self.scale_lin[b])
        loc = scale * (u @ v.t()) + self.bias[b]
        std = F.softplus(self.std_lin[b]) + EPS
        return ZILN(self.zi_logits[b].expand_as(loc), loc, std)


class NBDataDecoder(DataDecoder):

    r"""
    Negative binomial data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.log_theta = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: torch.Tensor
    ) -> D.NegativeBinomial:
        scale = F.softplus(self.scale_lin[b])
        logit_mu = scale * (u @ v.t()) + self.bias[b]
        mu = F.softmax(logit_mu, dim=1) * l
        log_theta = self.log_theta[b]
        return D.NegativeBinomial(
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )
    

class StratifiedNBDataDecoder(DataDecoder):
    r"""
    Modified Negative binomial data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1, input_dim: int = 5, embedding_size: int = 50, n_nodes = 10000,
                 feature_masks: list = []) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.log_theta = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.input_dim = input_dim
        #self.query_layers = []
        self.key_layers = []
        for i in range(input_dim):
            self.query_layers.append(torch.nn.Linear(embedding_size, embedding_size))
            self.key_layers.append(torch.nn.Linear(embedding_size, embedding_size))
        #self.query_layers = torch.nn.ModuleList(self.query_layers)
        self.key_layers = torch.nn.ModuleList(self.key_layers)
        self.feature_masks = feature_masks

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: torch.Tensor
    ) -> D.NegativeBinomial:
        mu_slices = []
        scale = F.softplus(self.scale_lin[b])
        log_theta = self.log_theta[b]
        for k in range(self.input_dim):
            strata_indices = self.feature_masks[k]
            scale_slice = scale[:, strata_indices]
            bias_slice = self.bias[b][:, strata_indices]
            #query = self.query_layers[k](u)
            query = u
            key = self.key_layers[k](v)
            decoded_strata = (query @ key.t())[:, :scale_slice.shape[1]]  # decode (ignoring excluded anchors at this strata)
            logit_mu = scale_slice * decoded_strata + bias_slice
            mu = F.softmax(logit_mu, dim=1) * l
            mu_slices.append(mu)

        mu = torch.concat(mu_slices, dim=1)
        return D.NegativeBinomial(
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )
    

class StratifiedZINBDataDecoder(DataDecoder):
    r"""
    Modified Zero-inflated negative binomial data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1, input_dim: int = 5, embedding_size: int = 50, n_nodes = 10000, dropout: float = 0.2,
                 feature_masks: list = [], strata_masks: list = [], shifted_additive: bool = False, use_activation: bool = False, use_attn: bool = False) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.log_theta = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.input_dim = input_dim
        #self.output_dropout = torch.nn.Dropout(p=dropout)
        self.query_layers = []
        self.key_convs = []
        self.key_conv_activations = []
        self.key_layers = []
        self.key_ln_layers = []
        self.attn_layers = []
        # self.value_layers = []
        # self.self_query_layers = []
        # self.self_key_layers = []
        # self.self_value_layers = []
        n_strata_features = len(feature_masks[0])
        self.self_ln = torch.nn.LayerNorm(embedding_size)
        for i in range(input_dim - 1):
            #self.query_layers.append(torch.nn.Linear(embedding_size, embedding_size))
            key_conv = torch.nn.Conv1d(embedding_size, embedding_size, kernel_size=2, dilation=i+1, padding='same', groups=embedding_size, bias=shifted_additive)
            key_conv_activation = torch.nn.LeakyReLU(negative_slope=0.2)
            # set init weights to 1/3 so we start off just taking feature combinations
            #torch.nn.init.constant_(key_conv.weight, 1/3)
            self.key_convs.append(key_conv)
            self.key_conv_activations.append(key_conv_activation)
            self.key_layers.append(torch.nn.Linear(embedding_size, embedding_size))
            #self.key_ln_layers.append(torch.nn.LayerNorm(embedding_size))
            self.attn_layers.append(LocalAttention(
                                    dim = embedding_size,
                                    window_size = input_dim,
                                    autopad = True,
                                    shared_qk = True))
            # self.value_layers.append(torch.nn.Linear(embedding_size, embedding_size))
            # self.self_query_layers.append(torch.nn.Linear(embedding_size, embedding_size))
            # self.self_key_layers.append(torch.nn.Linear(embedding_size, embedding_size))
            # self.self_value_layers.append(torch.nn.Linear(embedding_size, embedding_size))
        #self.query_layers = torch.nn.ModuleList(self.query_layers)
        self.key_layers = torch.nn.ModuleList(self.key_layers)
        self.key_convs = torch.nn.ModuleList(self.key_convs)
        self.key_conv_activations = torch.nn.ModuleList(self.key_conv_activations)
        #self.key_ln_layers = torch.nn.ModuleList(self.key_ln_layers)
        self.attn_layers = torch.nn.ModuleList(self.attn_layers)
        # self.value_layers = torch.nn.ModuleList(self.value_layers)
        # self.self_query_layers = torch.nn.ModuleList(self.self_query_layers)
        # self.self_key_layers = torch.nn.ModuleList(self.self_key_layers)
        # self.self_value_layers = torch.nn.ModuleList(self.self_value_layers)
        self.feature_masks = feature_masks
        self.strata_masks = strata_masks
        self.shifted_additive = shifted_additive
        self.use_activation = use_activation
        self.use_attn = use_attn
        self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.attn_norm = 1 / math.sqrt(embedding_size)
        self.embedding_size = embedding_size

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: torch.Tensor
    ) -> D.NegativeBinomial:
        mu_slices = []
        scale = F.softplus(self.scale_lin[b])
        log_theta = self.log_theta[b]
        #strata_start = 0
        for k in range(self.input_dim):
            strata_indices = self.strata_masks[k]
            feature_indices = self.feature_masks[k]
            scale_slice = scale[:, strata_indices]
            bias_slice = self.bias[b][:, strata_indices]
            #query = self.query_layers[k](u)
            #strata_embedding_size = int(self.embedding_size / self.input_dim)
            query = u
            #features = v[feature_indices, :]
            
            if k == 0:
                # first strata is reconstrcuted as normal (considered as self-loops)
                key = v
                #key = self.self_ln(key)
            else:
                # distal strata are reconstructed via inner product with their associated strata and a linear layer
                # maybe something like v * torch.roll(v, k, dims=1) ??
                # or consider embedding distance using something like torch.roll(v, k)
                
                #v = self.strata_attn(qk, qk, v)
                if self.use_attn:
                    qk = self.key_layers[k - 1](v)
                    key = self.attn_layers[k - 1](qk, qk, v)
                else:
                    key = self.key_convs[k - 1](v.t()).t()
                    if self.use_activation:
                        key = self.key_conv_activations[k - 1](key)
                #key = self.key_ln_layers[k - 1](key)
                #key = v * self.strata_key_masks[k]
                #key = self.key_layers[k](key)
            
            #key, _ = self.attn_layers[k](features, features, features)
            decoded_strata = (query @ key.t())[:, feature_indices]  # decode (ignoring excluded anchors at this strata)
            # self_query = self.self_query_layers[k](features)
            # self_key = self.self_key_layers[k](features)
            # #self_values = self.self_value_layers[k](values)
            # self_attention = (self_query @ self_key.t()) * self.attn_norm
            # self_attention = F.softmax(self_attention, dim=1)
            # # print(attn.shape)
            # # value = self.value_layers[k](v)
            # # print(value.shape)
            # decoded_strata = (values @ self_attention)
            #decoded_strata = self.output_dropout(decoded_strata)
            logit_mu = scale_slice * decoded_strata + bias_slice
            mu = F.softmax(logit_mu, dim=1) * l
            mu_slices.append(mu)

        mu = torch.concat(mu_slices, dim=1)  # because of this we need at least the strata to be sorted in the node embedding
        return ZINB(
            self.zi_logits[b].expand_as(mu),
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )
    

class ModifiedNBDataDecoder(DataDecoder):
    r"""
    Modified Negative binomial data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1, input_dim: int = 5, embedding_size: int = 50, n_nodes = 10000,
                 feature_masks: list = []) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.input_dim = input_dim
        self.out_features = out_features
        self.embedding_size = embedding_size
        self.n_nodes = n_nodes
        self.strata_intervals = list(range(0, self.embedding_size + 1, int(self.embedding_size / self.input_dim)))
        self.strata_intervals.append(-1)
        self.strata_intervals = [(self.strata_intervals[i], self.strata_intervals[i+1]) for i in range(len(self.strata_intervals) - 1)]
        if len(feature_masks) == 0:
            self.feature_intervals = list(range(0, out_features, self.n_nodes))
            self.feature_intervals.append(-1)
            self.feature_intervals = [(self.feature_intervals[i], self.feature_intervals[i+1]) for i in range(len(self.feature_intervals) - 1)]
        else:
            self.feature_masks = feature_masks
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.log_theta = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
        self, u: torch.Tensor, v: torch.Tensor, b: torch.Tensor, l: torch.Tensor
    ) -> D.NegativeBinomial:
        scale = F.softplus(self.scale_lin[b])

        mu_slices = []
        for k in range(self.input_dim):
            start_idx, end_idx = self.strata_intervals[k]
            #feat_start_idx, feat_end_idx = self.feature_intervals[k]
            strata_indices = self.feature_masks[k]
            u_slice = u[:, start_idx:end_idx]
            v_slice = v[:, start_idx:end_idx]
            #print(u_slice.shape, v_slice.shape)
            #print(scale[:, feat_start_idx:feat_end_idx].shape, self.bias[b][:, feat_start_idx:feat_end_idx].shape)
            scale_slice = scale[:, strata_indices]
            bias_slice = self.bias[b][:, strata_indices]
            constrained_decoded = (u_slice @ v_slice.t())[:, :scale_slice.shape[1]]
            
            #print(constrained_decoded.shape)
            # pad decoded matrix so dim 1 is the same size as feat_end_idx - feat_start_idx
            #constrained_decoded = F.pad(constrained_decoded, (0, max(0, scale_slice.shape[1] - constrained_decoded.shape[1]), 0, 0))
            #print(constrained_decoded.shape)
            logit_mu_slice = scale_slice * constrained_decoded + bias_slice
            mu_slice = F.softmax(logit_mu_slice, dim=1) * l
            mu_slices.append(mu_slice)

        # concat the slices to get the higher dimensional output
        mu = torch.concat(mu_slices, dim=1)
        #print(mu.shape)
        # pad the means so that dim 1 is the same size as out_features
        #mu = F.pad(mu, (0, max(0, self.out_features - mu.shape[1]), 0, 0))
        #print(mu.shape)

        log_theta = self.log_theta[b]
        return D.NegativeBinomial(
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )


class ZINBDataDecoder(NBDataDecoder):

    r"""
    Zero-inflated negative binomial data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> ZINB:
        scale = F.softplus(self.scale_lin[b])
        logit_mu = scale * (u @ v.t()) + self.bias[b]
        mu = F.softmax(logit_mu, dim=1) * l
        log_theta = self.log_theta[b]
        return ZINB(
            self.zi_logits[b].expand_as(mu),
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )
    

class ModifiedZINBDataDecoder(ModifiedNBDataDecoder):
    
        r"""
        Modified Zero-inflated negative binomial data decoder
    
        Parameters
        ----------
        out_features
            Output dimensionality
        n_batches
            Number of batches
        """
    
        def __init__(self, out_features: int, n_batches: int = 1, input_dim: int = 5, embedding_size: int = 50, n_nodes = 10000,
                     feature_masks: list = []) -> None:
            super().__init__(out_features, n_batches=n_batches, input_dim=input_dim, embedding_size=embedding_size, n_nodes=n_nodes,
                             feature_masks=feature_masks)
            self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))
    
        def forward(
            self, u: torch.Tensor, v: torch.Tensor, b: torch.Tensor, l: Optional[torch.Tensor]
        ) -> ZINB:
            scale = F.softplus(self.scale_lin[b])
    
            mu_slices = []
            for k in range(self.input_dim):
                start_idx, end_idx = self.strata_intervals[k]
                #feat_start_idx, feat_end_idx = self.feature_intervals[k]
                strata_indices = self.feature_masks[k]
                u_slice = u[:, start_idx:end_idx]
                v_slice = v[:, start_idx:end_idx]
                #print(u_slice.shape, v_slice.shape)
                #print(scale[:, feat_start_idx:feat_end_idx].shape, self.bias[b][:, feat_start_idx:feat_end_idx].shape)
                scale_slice = scale[:, strata_indices]
                bias_slice = self.bias[b][:, strata_indices]
                constrained_decoded = (u_slice @ v_slice.t())[:, :scale_slice.shape[1]]
                #print(constrained_decoded.shape)
                # pad decoded matrix so dim 1 is the same size as feat_end_idx - feat_start_idx
                #constrained_decoded = F.pad(constrained_decoded, (0, max(0, scale_slice.shape[1] - constrained_decoded.shape[1]), 0, 0))
                #print(constrained_decoded.shape)
                #print(scale_slice.shape, constrained_decoded.shape, bias_slice.shape)
                logit_mu_slice = scale_slice * constrained_decoded + bias_slice
                mu_slice = F.softmax(logit_mu_slice, dim=1) * l
                mu_slices.append(mu_slice)
    
            # concat the slices to get the higher dimensional output
            mu = torch.concat(mu_slices, dim=1)
            #print(mu.shape)
            # pad the means so that dim 1 is the same size as out_features
            mu = F.pad(mu, (0, max(0, self.out_features - mu.shape[1]), 0, 0))
            log_theta = self.log_theta[b]
            #print(mu.shape)
            return ZINB(
                self.zi_logits[b].expand_as(mu),
                log_theta.exp(),
                logits=(mu + EPS).log() - log_theta
            )



class Discriminator(torch.nn.Sequential, glue.Discriminator):

    r"""
    Modality discriminator

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def __init__(
            self, in_features: int, out_features: int, n_batches: int = 0,
            h_depth: int = 2, h_dim: Optional[int] = 256,
            dropout: float = 0.2
    ) -> None:
        self.n_batches = n_batches
        od = collections.OrderedDict()
        ptr_dim = in_features + self.n_batches
        for layer in range(h_depth):
            od[f"linear_{layer}"] = torch.nn.Linear(ptr_dim, h_dim)
            od[f"act_{layer}"] = torch.nn.LeakyReLU(negative_slope=0.2)
            od[f"dropout_{layer}"] = torch.nn.Dropout(p=dropout)
            ptr_dim = h_dim
        od["pred"] = torch.nn.Linear(ptr_dim, out_features)
        super().__init__(od)

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        if self.n_batches:
            b_one_hot = F.one_hot(b, num_classes=self.n_batches)
            x = torch.cat([x, b_one_hot], dim=1)
        return super().forward(x)


class Classifier(torch.nn.Linear):

    r"""
    Linear label classifier

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    """


class Prior(glue.Prior):

    r"""
    Prior distribution

    Parameters
    ----------
    loc
        Mean of the normal distribution
    std
        Standard deviation of the normal distribution
    """

    def __init__(
            self, loc: float = 0.0, std: float = 1.0
    ) -> None:
        super().__init__()
        loc = torch.as_tensor(loc, dtype=torch.get_default_dtype())
        std = torch.as_tensor(std, dtype=torch.get_default_dtype())
        self.register_buffer("loc", loc)
        self.register_buffer("std", std)

    def forward(self) -> D.Normal:
        return D.Normal(self.loc, self.std)


#-------------------- Network modules for independent GLUE ---------------------

class IndDataDecoder(DataDecoder):

    r"""
    Data decoder mixin that makes decoding independent of feature latent

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(  # pylint: disable=unused-argument
            self, in_features: int, out_features: int, n_batches: int = 1
    ) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.v = torch.nn.Parameter(torch.zeros(out_features, in_features))

    def forward(  # pylint: disable=arguments-differ
            self, u: torch.Tensor, b: torch.Tensor,
            l: Optional[torch.Tensor]
    ) -> D.Distribution:
        r"""
        Decode data from sample latent

        Parameters
        ----------
        u
            Sample latent
        b
            Batch index
        l
            Optional normalizer

        Returns
        -------
        recon
            Data reconstruction distribution
        """
        return super().forward(u, self.v, b, l)


class IndNormalDataDocoder(IndDataDecoder, NormalDataDecoder):
    r"""
    Normal data decoder independent of feature latent
    """


class IndZINDataDecoder(IndDataDecoder, ZINDataDecoder):
    r"""
    Zero-inflated normal data decoder independent of feature latent
    """


class IndZILNDataDecoder(IndDataDecoder, ZILNDataDecoder):
    r"""
    Zero-inflated log-normal data decoder independent of feature latent
    """


class IndNBDataDecoder(IndDataDecoder, NBDataDecoder):
    r"""
    Negative binomial data decoder independent of feature latent
    """


class IndZINBDataDecoder(IndDataDecoder, ZINBDataDecoder):
    r"""
    Zero-inflated negative binomial data decoder independent of feature latent
    """
