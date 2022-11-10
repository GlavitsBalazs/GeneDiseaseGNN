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

import copy
import functools
import os
from typing import Callable, Tuple, TypeAlias

import numpy as np
import scipy.sparse as sp
import torch
import tqdm
from scipy.sparse.linalg import LinearOperator
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType

from model import Prediction

SparseSolver: TypeAlias = Callable[..., Tuple[np.ndarray, int]]


def rwrh(data: HeteroData, lambda_=0.5, alpha=0.7, ppi_prune_threshold=0.4, pda_high_quality_threshold=0.1):
    # Yongjin Li, Jagdish C Patra: Genome-wide inferring gene-phenotype relationship by walking on the heterogeneous network
    # https://doi.org/10.1093/bioinformatics/btq108

    def prune_edges(data, edge_type: EdgeType, mask: Tensor):
        data[edge_type].edge_index = data[edge_type].edge_index[:, mask]
        data[edge_type].edge_attr = data[edge_type].edge_attr[mask]

    p_count, d_count = data['protein'].num_nodes, data['disease'].num_nodes

    ppi_weights = data['protein', 'associated_with', 'protein'].edge_attr.T[0]
    prune_edges(data, ('protein', 'associated_with', 'protein'), ppi_weights >= ppi_prune_threshold)

    pda_edge_scores = data['protein', 'associated_with', 'disease'].edge_attr.T[0]
    high_quality_mask = pda_edge_scores >= pda_high_quality_threshold
    prune_edges(data, ('protein', 'associated_with', 'disease'), high_quality_mask)
    prune_edges(data, ('disease', 'associated_with', 'protein'), high_quality_mask)

    data['protein', 'associated_with', 'disease'].edge_attr = torch.ones_like(
        data['protein', 'associated_with', 'disease'].edge_attr)

    data['disease', 'associated_with', 'protein'].edge_attr = torch.ones_like(
        data['disease', 'associated_with', 'protein'].edge_attr)

    a_p = sp.coo_matrix((data['protein', 'associated_with', 'protein'].edge_attr.T[0].numpy(),
                         data['protein', 'associated_with', 'protein'].edge_index.numpy()),
                        shape=(p_count, p_count))

    a_d = sp.coo_matrix((data['disease', 'associated_with', 'disease'].edge_attr.T[0].numpy(),
                         data['disease', 'associated_with', 'disease'].edge_index.numpy()),
                        shape=(d_count, d_count))

    b = sp.coo_matrix((data['protein', 'associated_with', 'disease'].edge_attr.T[0].numpy(),
                       data['protein', 'associated_with', 'disease'].edge_index.numpy()),
                      shape=(p_count, d_count))

    a_p_deg = np.asarray(a_p.sum(axis=0)).flatten()
    b_d_deg = np.asarray(b.sum(axis=0)).flatten()  # b_d_deg[j] = sum_i b[i,j]
    b_p_deg = np.asarray(b.sum(axis=1)).flatten()  # b_p_deg[i] = sum_j b[i,j]
    a_d_deg = np.asarray(a_d.sum(axis=0)).flatten()

    m_p = a_p.tocsr(copy=True)
    m_pd = b.tocsr(copy=True)
    m_dp = b.T.tocsr(copy=True)
    m_d = a_d.tocsr(copy=True)

    for i in range(p_count):
        m_p[i, :] *= 1 / a_p_deg[i] if a_p_deg[i] > 0 else 0
        m_p[i, :] *= (1 - lambda_) if b_p_deg[i] > 0 else 1
    for i in range(p_count):
        m_pd[i, :] *= lambda_ / b_p_deg[i] if b_p_deg[i] > 0 else 0
    for j in range(d_count):
        m_dp[j, :] *= lambda_ / b_d_deg[j] if b_d_deg[j] > 0 else 0
    for i in range(d_count):
        m_d[i, :] *= 1 / a_d_deg[i] if a_d_deg[i] > 0 else 0
        m_d[i, :] *= (1 - lambda_) if b_d_deg[i] > 0 else 1

    m: sp.csc_matrix = sp.bmat([[m_p, m_pd], [m_dp, m_d]], format="csc")
    m = m.T  # here I deviate from Li & Patra in that my transition matrix is left stochastic

    matvec_progress = tqdm.tqdm(total=d_count)

    def _lazy_sparse_inverse_matvec(y: np.ndarray, mat: sp.csc_matrix, solver: SparseSolver, scale: float):
        x, exit_code = solver(mat, y * scale)
        matvec_progress.update()  # remove this if not needed
        assert exit_code == 0, f"SparseSolver failed. {exit_code=}"
        return x

    def lazy_sparse_inverse(mat: sp.spmatrix, solver: SparseSolver, scale: float = 1.0) -> LinearOperator:
        matvec = functools.partial(_lazy_sparse_inverse_matvec, mat=mat.tocsc(), scale=scale, solver=solver)
        # noinspection PyArgumentList
        return LinearOperator(mat.shape, matvec, dtype=np.float64)

    def rwr(transition_matrix: sp.spmatrix, alpha: float, solver: SparseSolver):
        system_matrix = sp.identity(transition_matrix.shape[0]) - alpha * transition_matrix
        return lazy_sparse_inverse(system_matrix, solver, 1 - alpha)

    kernel = rwr(m, alpha, solver=functools.partial(sp.linalg.lgmres, atol=1e-5))
    scores_from_diseases = kernel @ (np.eye(m.shape[0], m.shape[1])[:, p_count:])
    scores_protein2disease = scores_from_diseases[:p_count, :]
    return scores_protein2disease


def get_rwrh_pred(data: HeteroData, cache_path='rwrh_pred_lambda_0.5_alpha_0.7.npz'):
    if not os.path.exists(cache_path):
        rwrh_pred = rwrh(copy.copy(data))
        np.savez_compressed(cache_path, rwrh_pred)
    else:
        rwrh_pred = np.load(cache_path)['arr_0']

    return Prediction(rwrh_pred, data['protein'].label, data['disease'].label,
                      gene_names=data['protein'].name, phenotype_names=data['disease'].name)
