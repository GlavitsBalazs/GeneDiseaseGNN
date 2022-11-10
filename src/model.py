# Copyright (C) 2022 Balázs Róbert Glávits

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# long with this program.  If not, see <https://www.gnu.org/licenses/>.

import abc
import math
import random
import time
from copy import copy
from typing import Callable, Sequence, Mapping, Tuple, Union, Optional, Type, NamedTuple, Collection

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import Tensor
from torch.nn import Linear, Module
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import HeteroData
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj, EdgeType, NodeType
import torch_geometric.utils as pyg_utils


class Encoder(Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x_dict: Mapping[NodeType, Tensor],
                edge_index_dict: Mapping[EdgeType, Adj],
                **conv_kwargs) -> Mapping[NodeType, Tensor]:
        raise NotImplementedError


class Decoder(Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, left_embeddings: Tensor, right_embeddings: Tensor) -> Tensor:
        raise NotImplementedError


class NegativeEdgeSampler(abc.ABC):
    def __init__(self, edge_index: Tensor,
                 num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
                 num_neg_samples: Optional[int] = None):
        pass

    @abc.abstractmethod
    def _sample(self) -> Tensor:
        raise NotImplementedError

    def sample(self) -> Mapping[EdgeType, Tensor]:
        neg_edge_index = self._sample()
        return {('protein', 'associated_with', 'disease'): neg_edge_index}

    @classmethod
    def create_for_data(cls, message: HeteroData, supervision: HeteroData, neg_ratio=1.0):
        num_nodes = (message['protein'].num_nodes, message['disease'].num_nodes)
        excluded_edges = torch.hstack((supervision['protein', 'disease'].edge_index,
                                       message['protein', 'disease'].edge_index))
        num_neg_samples = int(supervision['protein', 'disease'].edge_index.size(1) * neg_ratio)
        return cls(excluded_edges, num_nodes, num_neg_samples)


class Model(Module, abc.ABC):

    @property
    @abc.abstractmethod
    def encoder(self) -> Encoder:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def decoder(self) -> Decoder:
        raise NotImplementedError

    def encode(self, x_dict: Mapping[NodeType, Tensor], message_edge_index_dict: Mapping[EdgeType, Adj],
               message_edge_attr_dict: Mapping[EdgeType, Tensor]) -> Mapping[NodeType, Tensor]:
        protein_weights = {
            ('protein', 'associated_with', 'protein'):
                message_edge_attr_dict[('protein', 'associated_with', 'protein')].T[0]
        }
        return self.encoder(x_dict, message_edge_index_dict, edge_weight_dict=protein_weights)

    def decode(self, embedding_dict: Mapping[NodeType, Tensor],
               supervision_edge_index_dict: Mapping[EdgeType, Adj]) -> Tensor:
        row, col = supervision_edge_index_dict['protein', 'associated_with', 'disease']
        protein_embeddings, disease_embeddings = embedding_dict['protein'][row], embedding_dict['disease'][col]
        return self.decoder(protein_embeddings, disease_embeddings)

    def forward(self, x_dict: Mapping[NodeType, Tensor],
                message_edge_index_dict: Mapping[EdgeType, Adj],
                message_edge_attr_dict: Mapping[EdgeType, Tensor],
                supervision_edge_index_dict: Mapping[EdgeType, Adj]) -> Tensor:
        embedding_dict = self.encode(x_dict, message_edge_index_dict, message_edge_attr_dict)
        return self.decode(embedding_dict, supervision_edge_index_dict)

    def forward_pos_neg(self, x_dict: Mapping[NodeType, Tensor],
                        message_edge_index_dict: Mapping[EdgeType, Adj],
                        message_edge_attr_dict: Mapping[EdgeType, Tensor],
                        supervision_edge_index_dict: Mapping[EdgeType, Adj],
                        neg_sampler: NegativeEdgeSampler):
        # When calling `forward` twice, the embedding vectors also have to be computed twice.
        # Here instead we can reuse them.
        embedding_dict = self.encode(x_dict, message_edge_index_dict, message_edge_attr_dict)

        pred_pos = self.decode(embedding_dict, supervision_edge_index_dict)

        neg_edge_index_dict = neg_sampler.sample()
        pred_neg = self.decode(embedding_dict, neg_edge_index_dict)

        y_pred = torch.cat([pred_pos, pred_neg])
        y_true = torch.cat([torch.ones_like(pred_pos), torch.zeros_like(pred_neg)])
        return y_pred, y_true

    @torch.no_grad()
    def predict_raw(self, x_dict: Mapping[NodeType, Tensor],
                    message_edge_index_dict: Mapping[EdgeType, Adj],
                    message_edge_attr_dict: Mapping[EdgeType, Tensor]):
        embedding_dict = self.encode(x_dict, message_edge_index_dict, message_edge_attr_dict)
        p_count, d_count = embedding_dict['protein'].size(0), embedding_dict['disease'].size(0)
        res = torch.empty((p_count, d_count))
        for p in range(p_count):
            all_edges_from_protein = {
                ('protein', 'associated_with', 'disease'): torch.tensor(([p] * d_count, range(d_count)))
            }
            res[p, :] = self.decode(embedding_dict, all_edges_from_protein)
        return res

    def predict(self, message: HeteroData):
        scores = self.predict_raw(message.x_dict, message.edge_index_dict, message.edge_attr_dict)
        scores = torch.sigmoid(scores)
        scores = scores.numpy()
        return Prediction(scores, message['protein'].label, message['disease'].label,
                          gene_names=message['protein'].name, phenotype_names=message['disease'].name)


class Prediction(NamedTuple):
    scores: np.ndarray  # shape is (len(genes), len(diseases))
    gene_ids: Sequence[str]
    phenotype_ids: Sequence[str]
    gene_names: Optional[Sequence[str]] = None
    phenotype_names: Optional[Sequence[str]] = None

    def save(self, path):
        npz_fields = {'scores': self.scores, 'gene_ids': self.gene_ids, 'phenotype_ids': self.phenotype_ids}
        if self.gene_names is not None:
            npz_fields['gene_names'] = self.gene_names
        if self.phenotype_names is not None:
            npz_fields['phenotype_names'] = self.phenotype_names
        np.savez_compressed(path, **npz_fields)

    @classmethod
    def load(cls, path):
        npz = np.load(path)
        scores = npz['scores']
        gene_ids = npz['gene_ids']
        phenotype_ids = npz['phenotype_ids']
        gene_names = npz.get('gene_names')
        phenotype_names = npz.get('phenotype_names')
        return cls(scores, gene_ids, phenotype_ids, gene_names, phenotype_names)


def hinge_loss(y_pred, y_true, margin=0.1):
    y_pos = torch.sigmoid(y_pred[y_true == 1])
    y_neg = torch.sigmoid(y_pred[y_true == 0])
    return torch.mean(F.relu(y_neg - y_pos + margin))


def train(model: Model, optimizer: Optimizer, message: HeteroData, supervision: HeteroData,
          sampler: NegativeEdgeSampler,
          log: SummaryWriter | None = None, log_global_step: int | None = None):
    t0 = time.perf_counter()
    model.train()
    optimizer.zero_grad()
    y_pred, y_true = model.forward_pos_neg(message.x_dict, message.edge_index_dict, message.edge_attr_dict,
                                           supervision.edge_index_dict, sampler)
    loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
    loss.backward()
    optimizer.step()
    loss_item = loss.item()
    t1 = time.perf_counter()

    if log is not None:
        log.add_scalar('loss/train', loss_item, log_global_step)
        log.add_scalar('time/train', t1 - t0, log_global_step)
        log.flush()

    return loss_item


@torch.no_grad()
def evaluate(model: Model, message: HeteroData, supervision: HeteroData, sampler: NegativeEdgeSampler,
             log: SummaryWriter | None = None, log_global_step: int | None = None, log_group='val'):
    t0 = time.perf_counter()
    model.eval()
    y_pred, y_true = model.forward_pos_neg(message.x_dict, message.edge_index_dict, message.edge_attr_dict,
                                           supervision.edge_index_dict, sampler)
    loss = F.binary_cross_entropy_with_logits(y_pred, y_true)

    y_pred = torch.sigmoid(y_pred)
    loss_item = loss.item()
    y_true, y_pred = y_true.numpy(), y_pred.numpy()

    roc_auc = roc_auc_score(y_true, y_pred)
    avep = average_precision_score(y_true, y_pred)

    bin_y_pred = (y_pred > 0.5).astype(float)
    bin_roc_auc = roc_auc_score(y_true, bin_y_pred)
    bin_avep = average_precision_score(y_true, bin_y_pred)
    t1 = time.perf_counter()

    metrics = dict()
    metrics['loss'] = loss_item
    metrics['time'] = t1 - t0
    metrics['roc_auc'] = roc_auc
    metrics['roc_auc_bin'] = bin_roc_auc
    metrics['avep'] = avep
    metrics['avep_bin'] = bin_avep

    if log is not None:
        for k, v in metrics.items():
            log.add_scalar(f'{k}/{log_group}', v, log_global_step)
        log.add_pr_curve(f'pr/{log_group}', y_true, y_pred, log_global_step)
        log.add_pr_curve(f'bin_pr/{log_group}', y_true, bin_y_pred, log_global_step)
        log.flush()

    return loss_item, metrics


class InnerProductDecoder(Decoder):
    def forward(self, left_embeddings: Tensor, right_embeddings: Tensor):
        return (left_embeddings * right_embeddings).sum(dim=1)


class CosineSimilarityDecoder(Decoder):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, left_embeddings: Tensor, right_embeddings: Tensor):
        if self.normalize:
            left_embeddings = F.normalize(left_embeddings, p=2, dim=1)
            right_embeddings = F.normalize(right_embeddings, p=2, dim=1)
        cos_sim = (left_embeddings * right_embeddings).sum(dim=1)
        prob = (cos_sim + 1) / 2  # take it to [0,1] from [-1,1]
        # logits = torch.log(prob / (1 - prob))
        return prob


class MLPDecoder(Decoder):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.lin1 = Linear(2 * in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, left_embeddings: Tensor, right_embeddings: Tensor) -> Tensor:
        z = torch.cat([left_embeddings, right_embeddings], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


# Ideas for further development:
# dedicom decoder, distmult decoder, bilinear decoder
# skip connections
# different dropout or activation between layers
# deeper MLPs (GINs)
# attention

class HeteroConvEncoder(Encoder):
    def __init__(self, dim: Sequence[int],
                 node_types: Sequence[NodeType],
                 conv_factory: Callable[[int], Module],
                 use_pre_mlp=False,
                 use_inner_mlp=False,
                 use_batch_norm=False,
                 dropout=0.0,
                 act='relu',
                 use_post_mlp=False,
                 normalize=False):
        super().__init__()
        self.dim = dim

        self.node_types = node_types

        self.use_pre_mlp = use_pre_mlp
        if self.use_pre_mlp:
            self.pre_mlp = torch.nn.ModuleDict()
            for t in self.node_types:
                # issue: this dim must match the input dimension
                # though the first conv layer has lazy input dimension
                self.pre_mlp[t] = self.mlp_factory(dim[0])

        self.conv_layers = torch.nn.ModuleList()
        for d in self.dim:
            self.conv_layers.append(conv_factory(d))

        self.use_inner_mlp = use_inner_mlp
        if self.use_inner_mlp:
            self.inner_mlp_layers = torch.nn.ModuleList()
            for d in self.dim[:-1]:
                hetero_mlp = torch.nn.ModuleDict()
                for t in self.node_types:
                    hetero_mlp[t] = self.mlp_factory(d)
                self.inner_mlp_layers.append(hetero_mlp)
        else:
            self.inner_mlp_layers = [None for _ in self.dim[:-1]]
        self.inner_mlp_layers: Sequence[Mapping[NodeType, Module]]

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn_layers = torch.nn.ModuleList()
            for d in self.dim[:-1]:
                hetero_bn = torch.nn.ModuleDict()
                for t in self.node_types:
                    hetero_bn[t] = torch.nn.BatchNorm1d(d)
                self.bn_layers.append(hetero_bn)
        else:
            self.bn_layers = [None for _ in self.dim[:-1]]
        self.bn_layers: Sequence[Mapping[NodeType, Module]]

        self.dropout = dropout

        self.act_layers = torch.nn.ModuleList()
        for _ in self.dim[:-1]:
            hetero_act = torch.nn.ModuleDict()
            for t in self.node_types:
                hetero_act[t] = activation_resolver(act)
            self.act_layers.append(hetero_act)
        self.act_layers: Sequence[Mapping[NodeType, Module]]

        self.use_post_mlp = use_post_mlp
        if self.use_post_mlp:
            self.post_mlp = torch.nn.ModuleDict()
            for t in self.node_types:
                self.post_mlp[t] = self.mlp_factory(self.dim[-1])

        self.normalize = normalize

    @staticmethod
    def mlp_factory(dim: int):
        return pyg_nn.MLP(channel_list=[dim, dim], act='relu', plain_last=True)

    def forward(self, x_dict: Mapping[NodeType, Tensor],
                edge_index_dict: Mapping[EdgeType, Adj],
                **conv_kwargs) -> Mapping[NodeType, Tensor]:

        self.conv_layers: Sequence[pyg_nn.HeteroConv]
        self.inner_mlp_layers: Sequence[Mapping[NodeType, Module]]
        self.bn_layers: Sequence[Mapping[NodeType, Module]]
        self.act_layers: Sequence[Mapping[NodeType, Module]]
        zipped_layers = zip(self.conv_layers[:-1], self.inner_mlp_layers, self.bn_layers, self.act_layers)

        if self.use_pre_mlp:
            x_dict = {nt: self.pre_mlp[nt](x) for nt, x in x_dict.items()}
        for conv, mlp_dict, bn_dict, act_dict in zipped_layers:
            x_dict = conv(x_dict, edge_index_dict, **conv_kwargs)
            for nt in self.node_types:
                nt: NodeType
                if self.use_inner_mlp:
                    x_dict[nt] = mlp_dict[nt](x_dict[nt])
                if self.use_batch_norm:
                    x_dict[nt] = bn_dict[nt](x_dict[nt])
                if self.dropout > 0.0:
                    x_dict[nt] = F.dropout(x_dict[nt], self.dropout, self.training)
                x_dict[nt] = act_dict[nt](x_dict[nt])
        x_dict = self.conv_layers[-1](x_dict, edge_index_dict, **conv_kwargs)
        if self.use_post_mlp:
            x_dict = {nt: self.post_mlp[nt](x) for nt, x in x_dict.items()}
        if self.normalize:
            x_dict = {nt: F.normalize(x, p=2, dim=1) for nt, x in x_dict.items()}
        return x_dict


class UniformNegativeEdgeSampler(NegativeEdgeSampler):
    def __init__(self, edge_index: Tensor, num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
                 num_neg_samples: Optional[int] = None):
        super().__init__(edge_index, num_nodes, num_neg_samples)
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.num_neg_samples = num_neg_samples

    def _sample(self) -> Tensor:
        return pyg_utils.negative_sampling(self.edge_index, self.num_nodes, self.num_neg_samples)


class WeightedNegativeEdgeSampler(NegativeEdgeSampler):
    def __init__(self, edge_index: Tensor, num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
                 num_neg_samples: Optional[int] = None):
        super().__init__(edge_index, num_nodes, num_neg_samples)
        self.num_nodes = num_nodes if num_nodes is not None else (len(torch.unique(edge_index[i])) for i in (0, 1))
        self.num_neg_samples = num_neg_samples

        degree = [torch.bincount(edge_index[i], minlength=num_nodes[i]).numpy() for i in (0, 1)]
        self.weights = [d / d.sum() for d in degree]

        self.edge_sets = [set() if d > 0 else None for d in degree[0]]
        for u, v in edge_index.T:
            self.edge_sets[u].add(v)

        # maybe also store a np.random.Generator instance

    def is_negative(self, u, v):
        # maybe use a tree instead of a hash table
        return v not in self.edge_sets[u]

    def _sample(self) -> Tensor:
        samples: list[Tuple[int, int]] = list()
        while len(samples) < self.num_neg_samples:
            num_candidates = int((self.num_neg_samples - len(samples)) * 1.5)
            candidates = [np.random.choice(a=self.num_nodes[i], size=num_candidates, replace=True, p=self.weights[i])
                          for i in (0, 1)]
            for i in range(num_candidates):
                u, v = candidate = (candidates[0][i], candidates[1][i])
                if self.is_negative(u, v) and len(samples) < self.num_neg_samples:
                    samples.append(candidate)
        return torch.tensor(samples).T


def split_index_list(dataset_idx: Sequence, ratios: Sequence[float], min_size: int = 0) -> list[Sequence]:
    assert math.isclose(sum(ratios), 1) and all(r > 0 for r in ratios)
    assert min_size * len(ratios) <= len(dataset_idx)
    sizes = [max(min_size, int(round(len(dataset_idx) * r))) for r in reversed(ratios[1:])]
    sizes.append(len(dataset_idx) - sum(sizes))
    sizes.reverse()
    sizes = np.cumsum([0] + sizes)
    return [dataset_idx[sizes[i - 1]:sizes[i]] for i in range(1, len(sizes))]


def transductive_link_prediction_split(data: HeteroData):
    all_idx = list(range(data['disease', 'protein'].num_edges))
    random.shuffle(all_idx)

    # copying the dataset is fine because it isn't too big
    # otherwise masks would have to be used

    data_split = {k1: {k2: copy(data) for k2 in ('message', 'supervision')} for k1 in ('train', 'val', 'test')}
    data_split: dict[str, dict[str, HeteroData]]
    data_split_idx = {k1: {k2: None for k2 in ('message', 'supervision')} for k1 in ('train', 'val', 'test')}
    data_split_idx: dict[str, dict[str, list[int]]]

    train_message_idx, train_supervision_idx, val_idx, test_idx = split_index_list(all_idx, [0.7, 0.1, 0.1, 0.1],
                                                                                   min_size=0)
    data_split_idx['train']['message'] = train_message_idx
    data_split_idx['train']['supervision'] = train_supervision_idx
    data_split_idx['val']['message'] = train_message_idx + train_supervision_idx
    data_split_idx['val']['supervision'] = val_idx
    data_split_idx['test']['message'] = train_message_idx + train_supervision_idx + val_idx
    data_split_idx['test']['supervision'] = test_idx

    for k1 in data_split:
        for k2 in data_split[k1]:
            ds = data_split[k1][k2]
            idx = data_split_idx[k1][k2]
            ds['disease', 'protein'].edge_index = ds['disease', 'protein'].edge_index.T[idx,].T
            ds['disease', 'protein'].edge_attr = ds['disease', 'protein'].edge_attr[idx,]
            ds['protein', 'disease'].edge_index = ds['protein', 'disease'].edge_index.T[idx,].T
            ds['protein', 'disease'].edge_attr = ds['protein', 'disease'].edge_attr[idx,]

    return data_split


def load_and_preprocess_data(path='protein_disease_hetero_graph.pt',
                             protein_principal_components=128, disease_principal_components=128,
                             ppi_prune_threshold=0.0, pda_high_quality_threshold=0.1):
    def prune_edges(data, edge_type: EdgeType, mask: Tensor):
        data[edge_type].edge_index = data[edge_type].edge_index[:, mask]
        data[edge_type].edge_attr = data[edge_type].edge_attr[mask]

    data: HeteroData = torch.load(path)

    data['protein'].x = data['protein'].x[:, :protein_principal_components]
    data['disease'].x = data['disease'].x[:, :disease_principal_components]

    if ppi_prune_threshold > 0.0:
        ppi_weights = data['protein', 'associated_with', 'protein'].edge_attr.T[0]
        prune_edges(data, ('protein', 'associated_with', 'protein'), ppi_weights >= ppi_prune_threshold)

    pda_edge_scores = data['protein', 'associated_with', 'disease'].edge_attr.T[0]
    high_quality_mask = pda_edge_scores >= pda_high_quality_threshold
    medium_quality_mask = (pda_edge_scores < pda_high_quality_threshold) & (pda_edge_scores > 0.01)
    low_quality_mask = pda_edge_scores == 0.01

    high_quality_data = copy(data)
    prune_edges(high_quality_data, ('protein', 'associated_with', 'disease'), high_quality_mask)
    prune_edges(high_quality_data, ('disease', 'associated_with', 'protein'), high_quality_mask)

    medium_quality_data = copy(data)
    prune_edges(medium_quality_data, ('disease', 'associated_with', 'protein'), medium_quality_mask)
    prune_edges(medium_quality_data, ('protein', 'associated_with', 'disease'), medium_quality_mask)

    low_quality_data = copy(data)
    prune_edges(low_quality_data, ('disease', 'associated_with', 'protein'), low_quality_mask)
    prune_edges(low_quality_data, ('protein', 'associated_with', 'disease'), low_quality_mask)

    data_split = transductive_link_prediction_split(high_quality_data)
    data_split['test_medium'] = dict()
    data_split['test_medium']['message'] = data_split['test']['message']
    data_split['test_medium']['supervision'] = medium_quality_data
    data_split['test_low'] = dict()
    data_split['test_low']['message'] = data_split['test']['message']
    data_split['test_low']['supervision'] = low_quality_data
    return data_split


def bedroc_score(y_true, y_pred, decreasing=True, alpha=20.0):
    """
    BEDROC metric implemented according to Truchon and Bayley.

    Copyright (C) 2015-2016 Rich Lewis <rl403@cam.ac.uk>
    License: 3-clause BSD

    The Boltzmann Enhanced Descrimination of the Receiver Operator
    Characteristic (BEDROC) score is a modification of the Receiver Operator
    Characteristic (ROC) score that allows for a factor of *early recognition*.

    References:
        The original paper by Truchon et al. is located at `10.1021/ci600426e
        <http://dx.doi.org/10.1021/ci600426e>`_.

    Args:
        y_true (array_like):
            Binary class labels. 1 for positive class, 0 otherwise.
        y_pred (array_like):
            Prediction values.
        decreasing (bool):
            True if high values of ``y_pred`` correlates to positive class.
        alpha (float):
            Early recognition parameter.

    Returns:
        float:
            Value in interval [0, 1] indicating degree to which the predictive
            technique employed detects (early) the positive class.
     """

    assert len(y_true) == len(y_pred), \
        'The number of scores must be equal to the number of labels'

    N = len(y_true)
    n = sum(y_true == 1)

    if decreasing:
        order = np.argsort(-y_pred)
    else:
        order = np.argsort(y_pred)

    m_rank = (y_true[order] == 1).nonzero()[0]
    s = np.sum(np.exp(-alpha * m_rank / N))
    r_a = n / N
    rand_sum = r_a * (1 - np.exp(-alpha)) / (np.exp(alpha / N) - 1)
    fac = r_a * np.sinh(alpha / 2) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * r_a))
    cte = 1 / (1 - np.exp(alpha * (1 - r_a)))
    return s * fac / rand_sum + cte


def apk(actual: Collection, predicted, k=10):
    """
    Computes the average precision at k.

    Copyright (C) 2012 Ben Hamner <ben@benhamner.com>

    This function computes the average precision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)
