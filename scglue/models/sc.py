r"""
GLUE component modules for single-cell omics data
"""

import collections
from abc import abstractmethod
from typing import Optional, Tuple, List, Union

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
        self.conv2 = GraphAttent(out_features, out_features)
        self.loc = torch.nn.Linear(out_features, out_features)
        self.std_lin = torch.nn.Linear(out_features, out_features)

    def forward(
            self, eidx: torch.Tensor, enorm: torch.Tensor, esgn: torch.Tensor
    ) -> D.Normal:
        ptr = self.conv(self.vrepr, eidx, enorm, esgn)
        ptr = F.selu(ptr)
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

    def __init__(self, 
                in_features: int, out_features: int,
                h_depth: int = 2, h_dim: int = 256,
                dropout: float = 0.2,
                downsample_min: float = 0.0,
                downsample_max: float = 1.0,
                strata_masks: list = [],
                use_conv=False):
        if use_conv:
            # ensure that the matrix can be halved as many times as the depth
            depth = 3
            pool_padding_len = len(strata_masks[0]) + (2 ** depth - len(strata_masks[0]) % 2 ** depth)
            in_features += (pool_padding_len - len(strata_masks[0])) * len(strata_masks)
        super().__init__(in_features, out_features, h_depth, h_dim, dropout, downsample_min, downsample_max)
        self.strata_masks = strata_masks
        self.use_conv = use_conv
        if use_conv:
            self.pool_padding_len = pool_padding_len
            self.conv_net = torch.nn.Sequential(
                torch.nn.Conv2d(1, 8, kernel_size=(3, 3), padding='same'),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 8, kernel_size=(3, 3), padding='same'),
                torch.nn.ReLU(),
                #torch.nn.BatchNorm2d(8),
                torch.nn.MaxPool2d((1, 2)),
                torch.nn.Conv2d(8, 16, kernel_size=(3, 3), padding='same'),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 16, kernel_size=(3, 3), padding='same'),
                torch.nn.ReLU(),
                #torch.nn.BatchNorm2d(16),
                torch.nn.MaxPool2d((1, 2)),
                torch.nn.Conv2d(16, 32, kernel_size=(3, 3), padding='same'),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, kernel_size=(3, 3), padding='same'),
                torch.nn.ReLU(),
                #torch.nn.BatchNorm2d(16),
                torch.nn.MaxPool2d((1, 2)),
                torch.nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same'),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 8, kernel_size=(3, 3), padding='same'),
            )
        

    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)

    def normalize(
            self, x: torch.Tensor, l: torch.Tensor
    ) -> torch.Tensor:
        return (x * (self.TOTAL_COUNT / l)).log1p()
        #return x.log1p()

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
        if self.use_conv:
            # reshape into 2D matrix
            x = ptr.view(-1, 1, len(self.strata_masks), len(self.strata_masks[0]))
            x = F.pad(x, (0, self.pool_padding_len - len(self.strata_masks[0])))
            ptr = self.conv_net(x)
            ptr = ptr.view(ptr.size(0), -1)
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std), l


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
    

class GEGLU(torch.nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)
    

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
                 feature_masks: list = [], strata_masks: list = [], shifted_additive: bool = False, use_activation: bool = False, use_attn: bool = False,
                 binarize: bool = False) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.log_theta = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.input_dim = input_dim
        self.query_layers = []
        self.key_convs = []
        self.key_conv_activations = []
        self.key_layers = []
        self.value_layers = []
        self.key_ln_layers = []
        self.attn_layers = []
        self.prenorm_layers = []
        self.postnorm_layers = []
        self.ff_layers = []
        self.ff_activations = []
        self.self_ln = torch.nn.LayerNorm(embedding_size)
        for i in range(input_dim - 1):
            key_conv = torch.nn.Conv1d(embedding_size, embedding_size, kernel_size=2, dilation=i+1, padding='same', groups=embedding_size, bias=shifted_additive)
            key_conv_activation = torch.nn.LeakyReLU(negative_slope=0.2)
            # set init weights to 1/3 so we start off just taking feature combinations
            #torch.nn.init.constant_(key_conv.weight, 1/3)
            self.key_convs.append(key_conv)
            self.key_conv_activations.append(key_conv_activation)
            self.key_layers.append(torch.nn.Linear(embedding_size, embedding_size, bias=False))
            self.value_layers.append(torch.nn.Linear(embedding_size, embedding_size, bias=False))
            self.attn_layers.append(LocalAttention(
                                    dim = embedding_size,
                                    window_size = input_dim * 20,
                                    autopad = True,
                                    shared_qk = True))
            self.prenorm_layers.append(torch.nn.LayerNorm(embedding_size))
            self.postnorm_layers.append(torch.nn.LayerNorm(embedding_size))
            self.ff_layers.append(torch.nn.Linear(embedding_size, embedding_size * 2, bias=False))
            self.ff_activations.append(GEGLU())
        self.key_layers = torch.nn.ModuleList(self.key_layers)
        self.value_layers = torch.nn.ModuleList(self.value_layers)
        self.key_convs = torch.nn.ModuleList(self.key_convs)
        self.key_conv_activations = torch.nn.ModuleList(self.key_conv_activations)
        self.attn_layers = torch.nn.ModuleList(self.attn_layers)
        self.prenorm_layers = torch.nn.ModuleList(self.prenorm_layers)
        self.postnorm_layers = torch.nn.ModuleList(self.postnorm_layers)
        self.ff_layers = torch.nn.ModuleList(self.ff_layers)
        self.ff_activations = torch.nn.ModuleList(self.ff_activations)
        self.feature_masks = feature_masks
        self.strata_masks = strata_masks
        self.shifted_additive = shifted_additive
        self.use_activation = use_activation
        self.use_attn = use_attn
        self.binarize = binarize
        if binarize:
            self.ber_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        else:
            self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.attn_norm = 1 / math.sqrt(embedding_size)
        self.embedding_size = embedding_size
        self.strata_weights = torch.nn.Parameter(torch.ones(input_dim))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: torch.Tensor
    ) -> D.NegativeBinomial:
        mu_slices = []
        scale = F.softplus(self.scale_lin[b])
        log_theta = self.log_theta[b]
        #strata_start = 0
        weights = F.softmax(self.strata_weights, dim=0)
        for k in range(self.input_dim):
            strata_indices = self.strata_masks[k]
            feature_indices = self.feature_masks[k]
            scale_slice = scale[:, strata_indices]
            bias_slice = self.bias[b][:, strata_indices]
            query = u
            
            if k == 0:
                # first strata is reconstrcuted as normal (considered as self-loops)
                key = v
            else:
                # distal strata are reconstructed via inner product with their associated strata and a linear layer
                # maybe something like v * torch.roll(v, k, dims=1) ??
                # or consider embedding distance using something like torch.roll(v, k)
                if self.use_attn:
                    prenorm = self.prenorm_layers[k - 1](v)
                    qk = self.key_layers[k - 1](prenorm)
                    v = self.value_layers[k - 1](prenorm)
                    key = v + self.attn_layers[k - 1](qk, qk, prenorm)
                    key = key + self.ff_activations[k - 1](self.ff_layers[k - 1](self.postnorm_layers[k - 1](key)))
                else:
                    key = self.key_convs[k - 1](v.t()).t()
                    if self.use_activation:
                        key = self.key_conv_activations[k - 1](key)
    
            decoded_strata = (query @ key.t())[:, feature_indices]  # decode (ignoring excluded anchors at this strata)
            logit_mu = scale_slice * decoded_strata + bias_slice
            if self.binarize:
                mu = logit_mu
            else:
                mu = F.softmax(logit_mu, dim=1) * l * weights[k]
            mu_slices.append(mu)

        mu = torch.concat(mu_slices, dim=1)  # because of this we need at least the strata to be sorted in the node embedding
        if self.binarize:
            return D.Bernoulli(logits=mu)
        return ZINB(
            self.zi_logits[b].expand_as(mu),
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
