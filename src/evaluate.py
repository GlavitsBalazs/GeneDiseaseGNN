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

import pickle
from collections import defaultdict
from typing import Any, Callable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric.utils as pyg_utils
import tqdm
from scipy import stats
from scipy.stats._stats_py import SpearmanrResult
from sklearn import metrics
from torch_geometric import seed_everything
from torch_geometric.data import HeteroData

import baseline
import best_model
import rwrh
from model import Prediction, apk, bedroc_score


def benjamini_hochberg(items: Sequence, p_val: Callable[[Any], float], significance_level=0.05):
    sorted_items = sorted(((p_val(x), x) for x in items), key=lambda x: x[0])
    j_max = max([0] + [j + 1 for j in range(len(items))
                       if sorted_items[j][0] <= (j + 1) / len(items) * significance_level])
    return [sorted_items[j][1] for j in range(j_max)]


def compare_with_gwasatlas(pred: Prediction, gwas2cuis, gwas2prot):
    all_gwas_prot_ids = np.array(gwas2prot[0])
    pred_prot_ids = {ensp: i for i, ensp in enumerate(pred.gene_ids)}
    pred_cuis = {cui: i for i, cui in enumerate(pred.phenotype_ids)}
    correlations: dict[Tuple[int, int], SpearmanrResult] = dict()
    for gwas, (trait, cuis) in gwas2cuis.items():
        gwas_raw = gwas2prot[1][gwas]
        gwas_sigs = -np.log10(gwas_raw[:, 1])
        gwas_prot_ids = all_gwas_prot_ids[gwas_raw[:, 0].astype(int)]
        pred_rows = np.array([pred_prot_ids[ensp] for ensp in gwas_prot_ids])
        for cui in cuis:
            if cui not in pred_cuis:
                continue
            col = pred_cuis[cui]
            selected_scores = pred.scores[pred_rows, col]
            r: SpearmanrResult = stats.spearmanr(gwas_sigs, selected_scores)
            correlations[(cui, gwas)] = r

    bh = benjamini_hochberg(list(correlations.items()), p_val=lambda x: x[1].pvalue)
    return bh


def ranked_eval(score_vector, positives, k=200):
    ranking = list(np.flip(np.argsort(score_vector)))
    pos_ranks = {pos: ranking.index(pos) for pos in positives}
    if any(pos_ranks):
        avep = np.mean([(tp + 1) / (pp + 1) for tp, pp in enumerate(sorted(pos_ranks.values()))])
        pos_topk = np.isin(ranking[:k], list(pos_ranks.keys()))
        discounted_rankings = np.cumsum(pos_topk) / (1 + np.arange(len(pos_topk)))
        apk = np.sum(discounted_rankings[pos_topk,]) / min(len(pos_topk), len(pos_ranks))
        mrr = np.mean([1 / (r + 1) for r in pos_ranks.values()])
        return avep, apk, mrr, len(pos_ranks)
    return None


def binary_eval(pred: Prediction, disgenet_edge_index):
    all_edges = np.hstack(tuple(disgenet_edge_index.values()))
    res_by_k = dict()
    for k in disgenet_edge_index.keys():
        pos_edges = disgenet_edge_index[k]
        neg_edges = pyg_utils.negative_sampling(
            torch.from_numpy(all_edges), pred.scores.shape, pos_edges.shape[1]
        ).numpy()
        y_pos = pred.scores[pos_edges[0], pos_edges[1]]
        y_neg = pred.scores[neg_edges[0], neg_edges[1]]
        y_pred = np.hstack((y_pos, y_neg))
        y_true = np.hstack((np.ones(pos_edges.shape[1]), np.zeros(neg_edges.shape[1])))

        roc_sc = metrics.roc_auc_score(y_true, y_pred)
        aupr_sc = metrics.average_precision_score(y_true, y_pred)

        bedroc_sc = bedroc_score(y_true, y_pred)

        actual = list(np.where(y_true == 1)[0])
        predicted = [i for i, score in sorted(enumerate(y_pred), key=lambda x: x[1], reverse=True)]
        apk_sc = apk(actual, predicted, k=200)

        bin_y_pred = (y_pred > 0.5).astype(float)
        bin_roc_auc = metrics.roc_auc_score(y_true, bin_y_pred)
        bin_avep = metrics.average_precision_score(y_true, bin_y_pred)

        res_by_k[k] = (roc_sc, aupr_sc, bedroc_sc, apk_sc, bin_roc_auc, bin_avep)
    return res_by_k


def load_disgenet_edge_index(path='protein_disease_hetero_graph.pt'):
    # We already preprocessed it. No need to read the sqlite db.
    data: HeteroData = torch.load(path)
    pda_edge_scores = data['protein', 'associated_with', 'disease'].edge_attr.T[0]
    high_quality_edges = data['protein', 'associated_with', 'disease'].edge_index[:, pda_edge_scores >= 0.1].numpy()
    medium_quality_mask = (pda_edge_scores < 0.1) & (pda_edge_scores > 0.01)
    medium_quality_edges = data['protein', 'associated_with', 'disease'].edge_index[:, medium_quality_mask].numpy()
    low_quality_edges = data['protein', 'associated_with', 'disease'].edge_index[:, pda_edge_scores == 0.01].numpy()

    disgenet_edge_index = {'high': high_quality_edges, 'medium': medium_quality_edges, 'low': low_quality_edges}
    return disgenet_edge_index


def ranked_eval_with_disgenet(pred: Prediction, disgenet_edge_index):
    dis2prot = dict()
    for k in disgenet_edge_index.keys():
        dis2prot[k] = defaultdict(set)
        for prot, dis in disgenet_edge_index[k].T:
            dis2prot[k][dis].add(prot)

    scores = {k: dict() for k in disgenet_edge_index.keys()}

    for i, cui in tqdm.tqdm(enumerate(pred.phenotype_ids), total=len(pred.phenotype_ids)):
        rwr_scores = pred.scores[:, i]
        for k in disgenet_edge_index.keys():
            sc = ranked_eval(rwr_scores, dis2prot[k][i])
            if sc is not None:
                scores[k][cui] = sc

    for k in scores:
        scores[k] = sorted(scores[k].items(), key=lambda x: x[1][0], reverse=True)
    return scores


def edge_set_difference(a, b):
    return np.array(list(set(tuple(x) for x in a.T) - set(tuple(x) for x in b.T))).T


def main():
    seed_everything(0)

    best_model_timestamp = "2022-10-29T12:30:30+00:00"
    baseline_timestamp = "2022-10-29T18:59:27+00:00"

    # These three files are to be generated by dataset.py
    dataset_path = 'protein_disease_hetero_graph.pt'
    with open('gwas2cuis.pickle', "rb") as f:
        gwas2cuis = pickle.load(f)
        gwas2cuis: dict[int, Tuple[list[int], dict[str, float]]]
    with open('gwas2prot.pickle', "rb") as f:
        gwas2prot = pickle.load(f)
        gwas2prot: Tuple[list[str], dict[int, np.ndarray]]

    gnn_pred, gnn_training_edges = best_model.best_prediction(best_model_timestamp)
    print("GNN gwas")
    gwas_gnn = compare_with_gwasatlas(gnn_pred, gwas2cuis, gwas2prot)
    disgenet_edge_index_gnn = load_disgenet_edge_index()
    disgenet_edge_index_gnn['high'] = edge_set_difference(disgenet_edge_index_gnn['high'], gnn_training_edges)
    print("GNN ranked")
    ranked_gnn = ranked_eval_with_disgenet(gnn_pred, disgenet_edge_index_gnn)
    bin_gnn = binary_eval(gnn_pred, disgenet_edge_index_gnn)

    mlp_pred, mlp_training_edges = baseline.best_prediction(baseline_timestamp)
    print("MLP gwas")
    gwas_mlp = compare_with_gwasatlas(mlp_pred, gwas2cuis, gwas2prot)
    disgenet_edge_index_mlp = load_disgenet_edge_index(dataset_path)
    disgenet_edge_index_mlp['high'] = edge_set_difference(disgenet_edge_index_mlp['high'], mlp_training_edges)
    print("MLP ranked")
    ranked_mlp = ranked_eval_with_disgenet(mlp_pred, disgenet_edge_index_mlp)
    bin_mlp = binary_eval(mlp_pred, disgenet_edge_index_mlp)

    data: HeteroData = torch.load(dataset_path)
    disgenet_edge_index = load_disgenet_edge_index(dataset_path)
    rwrh_pred = rwrh.get_rwrh_pred(data, cache_path='rwrh_pred_lambda_0.5_alpha_0.7.npz')
    print("RWRH gwas")
    gwas_rwrh = compare_with_gwasatlas(rwrh_pred, gwas2cuis, gwas2prot)
    print("RWRH ranked")
    ranked_rwrh = ranked_eval_with_disgenet(rwrh_pred, disgenet_edge_index)
    bin_rwrh = binary_eval(rwrh_pred, disgenet_edge_index)

    random_pred = Prediction(np.random.uniform(0, 1, (len(data['protein'].label), len(data['disease'].label))),
                             data['protein'].label, data['disease'].label,
                             gene_names=data['protein'].name, phenotype_names=data['disease'].name)
    print("Random gwas")
    gwas_rand = compare_with_gwasatlas(random_pred, gwas2cuis, gwas2prot)
    print("Random ranked")
    ranked_rand = ranked_eval_with_disgenet(random_pred, disgenet_edge_index)
    bin_rand = binary_eval(random_pred, disgenet_edge_index)

    print("\n\nBinary metrics:\n")
    print("roc_sc, aupr_sc, bedroc_sc, apk_sc, bin_roc_auc, bin_avep")
    for i, bin_eval in enumerate((bin_gnn, bin_mlp, bin_rwrh, bin_rand)):
        print(("GNN", "MLP", "RWRH", "Random")[i])
        for q in ("high", "medium", "low"):
            print("\t".join((q,) + tuple(f"{x:.3f}" for x in bin_eval[q])))

    print("\n\nGWAS metrics:\n")
    pred_cuis = set(data['disease'].label)
    pred_gwas2cui = list()
    for gwas, (trait, cuis) in gwas2cuis.items():
        for cui in cuis:
            if cui not in pred_cuis:
                continue
            pred_gwas2cui.append((gwas, cui))
    # cui2name = dict(zip(data['disease'].label, data['disease'].name))
    gwas2name = {id_: x[0][0] for id_, x in gwas2cuis.items()}

    print("All phenotypes with prediction:", len(pred_cuis))
    print("All GWA studies: ", len(gwas2name))
    print("Matches of GWA studies and predicted phenotypes", len(pred_gwas2cui))
    print("GWA studies with prediction", len({g for g, c in pred_gwas2cui}))
    print("Phenotypes with prediction and GWA studies ", len({c for g, c in pred_gwas2cui}))
    print("\n Matches, Unique GWAS, Unique phenotypes")
    for i, gwa in enumerate((gwas_gnn, gwas_mlp, gwas_rwrh, gwas_rand)):
        print(("GNN", "MLP", "RWRH", "Random")[i])
        print(len(gwa), len({g for (c, g), sr in gwa}), len({c for (c, g), sr in gwa}))

    print("\n\nRanked metrics:\n")
    fig, axes = plt.subplots(2, 3, figsize=(3 * 5, 4), sharex=True)
    for i, (ax1, ax2), name in zip(range(3), axes.T, ("AveP", "AP@200", "MRR")):
        fig.subplots_adjust(hspace=0.05)
        for ax in (ax1, ax2):
            ax.hist([
                [x[1][i] for x in ranked_gnn['high']],
                [x[1][i] for x in ranked_mlp['high']]
            ], 15, log=False, range=(0, 1))

        # zoom-in / limit the view to different portions of the data
        ax1.set_ylim(65, 4500)  # outliers only
        ax2.set_ylim(0, 65)  # most of the data

        # hide the spines between ax and ax2
        ax1.spines.bottom.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()

        ax1.legend(["GNN", "MLP"])

        # see https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
        ax2.set_xlabel(name)
        ax1.set_title(f'{name} per phenotype')
    axes[1, 0].set_ylabel('Number of phenotypes')
    fig.suptitle("Ranked evaluation of all phenotypes over high quality DisGeNet edges")
    plt.savefig('ranked_evaluation_high.pdf', bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(3 * 5, 4), sharex=True)
    for i, (ax1, ax2), name in zip(range(3), axes.T, ("AveP", "AP@200", "MRR")):
        fig.subplots_adjust(hspace=0.05)
        for ax in (ax1, ax2):
            ax.hist([
                [x[1][i] for x in ranked_gnn['medium']],
                [x[1][i] for x in ranked_mlp['medium']],
                [x[1][i] for x in ranked_rwrh['medium']],
            ], 15, log=False, range=(0, 1))

        # zoom-in / limit the view to different portions of the data
        ax1.set_ylim(90, 3500)  # outliers only
        ax2.set_ylim(0, 90)  # most of the data

        # hide the spines between ax and ax2
        ax1.spines.bottom.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()

        ax1.legend(["GNN", "MLP", "RWRH"])

        # see https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
        ax2.set_xlabel(name)
        ax1.set_title(f'{name} per phenotype')
    axes[1, 0].set_ylabel('Number of phenotypes')
    fig.suptitle("Ranked evaluation of all phenotypes over medium quality DisGeNet edges")
    plt.savefig('ranked_evaluation_medium.pdf', bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(3 * 5, 4), sharex=True)
    for i, (ax1, ax2), name in zip(range(3), axes.T, ("AveP", "AP@200", "MRR")):
        fig.subplots_adjust(hspace=0.05)
        for ax in (ax1, ax2):
            ax.hist([
                [x[1][i] for x in ranked_gnn['low']],
                [x[1][i] for x in ranked_mlp['low']],
                [x[1][i] for x in ranked_rwrh['low']],
            ], 15, log=False, density=False, range=(0, 1))

        # zoom-in / limit the view to different portions of the data
        ax1.set_ylim(75, 4700)  # outliers only
        ax2.set_ylim(0, 75)  # most of the data
        # ax1.set_ylim(1, 20)  # outliers only
        # ax2.set_ylim(0, 1)  # most of the data

        # hide the spines between ax and ax2
        ax1.spines.bottom.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()

        ax1.legend(["GNN", "MLP", "RWRH"])

        # see https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
        ax2.set_xlabel(name)
        # ax2.set(xticks=np.linspace(0,1,9), xlim=[0, 1])
        ax1.set_title(f'{name} per phenotype')
    axes[1, 0].set_ylabel('Number of phenotypes')
    fig.suptitle("Ranked evaluation of all phenotypes over low quality DisGeNet edges")
    plt.savefig('ranked_evaluation_low.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
