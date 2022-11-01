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

import argparse
import csv
import gzip
import os.path
import pathlib
import pickle
import sqlite3
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Mapping, Any, Tuple, Sequence, Iterable, Container

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import torch
import tqdm
from sklearn.decomposition import TruncatedSVD
from torch_geometric.data import HeteroData

import csv
import gzip
import os
import sqlite3

from collections import defaultdict, Counter
import re
from typing import Sequence, Tuple, Mapping
import scipy.sparse as sp
import urllib.parse
import urllib.request
import numpy as np
import tqdm
from torch_geometric.data import HeteroData
import torch

import itertools


def load_protein_graph(input_path):
    """The input format is specific to the STRING database."""
    mat_row, mat_col, mat_data = [], [], []
    with gzip.open(input_path, 'rt') as protein_links:
        reader = csv.reader(protein_links, delimiter=' ')
        header = next(reader)
        protein1_col, protein2_col = header.index('protein1'), header.index('protein2')
        combined_score_col = header.index('combined_score')
        for row in tqdm.tqdm(reader, desc="Loading STRING protein links", unit='rows', total=11938498):
            protein1_ensembl_id = int(row[protein1_col][9:])
            protein2_ensembl_id = int(row[protein2_col][9:])
            combined_score = int(row[combined_score_col])
            mat_row.append(protein1_ensembl_id)
            mat_col.append(protein2_ensembl_id)
            mat_data.append(combined_score)
    mat_row = np.array(mat_row, dtype=np.int32)
    mat_col = np.array(mat_col, dtype=np.int32)
    mat_data = np.array(mat_data, dtype=np.int16)
    return sp.coo_matrix((mat_data, (mat_row, mat_col)))


def load_clusters_info(clusters_info):
    result = dict()
    with gzip.open(clusters_info, 'rt') as protein_links:
        reader = csv.reader(protein_links, delimiter='\t')
        header = next(reader)
        cluster_id_col, cluster_size_col = header.index('cluster_id'), header.index('cluster_size')
        best_described_by_col = header.index('best_described_by')
        for row in reader:
            cluster_id = row[cluster_id_col]
            cluster_size = int(row[cluster_size_col])
            best_described_by = row[best_described_by_col]
            result[cluster_id] = (cluster_size, best_described_by)
    return result


def load_clusters_tree(clusters_tree_path):
    clusters_tree = nx.DiGraph()
    with gzip.open(clusters_tree_path, 'rt') as protein_links:
        reader = csv.reader(protein_links, delimiter='\t')
        header = next(reader)
        child_cluster_id_col = header.index('child_cluster_id')
        parent_cluster_id_col = header.index('parent_cluster_id')
        for row in reader:
            child_cluster_id = row[child_cluster_id_col]
            parent_cluster_id = row[parent_cluster_id_col]
            clusters_tree.add_edge(parent_cluster_id, child_cluster_id)
    return clusters_tree


def load_clusters(clusters_proteins_path):
    res = defaultdict(list)
    with gzip.open(clusters_proteins_path, 'rt') as clusters_proteins:
        reader = csv.reader(clusters_proteins, delimiter='\t')
        header = next(reader)
        cluster_id_col, protein_id_col = header.index('cluster_id'), header.index('protein_id')
        for row in reader:
            cluster_id = row[cluster_id_col]
            protein_id = row[protein_id_col]
            res[protein_id].append(cluster_id)
    return res


def protein_ontology(clusters_tree_path, clusters_proteins_path):
    ontology = load_clusters_tree(clusters_tree_path)
    protein_enclosing_clusters = load_clusters(clusters_proteins_path)

    protein_msc = dict()  # most specific clusters
    # each protein belongs in many clusters but there is only one that's the smallest
    for protein, enclosing_clusters in protein_enclosing_clusters.items():
        enclosing_clusters_g: nx.DiGraph = ontology.subgraph(enclosing_clusters)
        leaves = [x for x in enclosing_clusters_g.nodes() if enclosing_clusters_g.out_degree[x] == 0]
        assert len(leaves) == 1
        protein_msc[protein] = leaves[0]

    for protein, cluster in protein_msc.items():
        ontology.add_edge(cluster, protein)
    return ontology


def bfs_layers(g: nx.DiGraph, sources: Iterable):
    frontier = {source: None for source in sources}
    while True:
        yield frontier
        new_frontier = set.union(*(set(g.successors(n)) for n in frontier.keys()))
        if len(new_frontier) == 0:
            break
        frontier = {n: set(g.predecessors(n)) for n in new_frontier}


def umls_ontology_subset(target_cuis: Iterable[str], mrrel_rrf_path):
    # Build an ontology (hierarchical tree of concepts) for each source vocabulary.
    hiers_by_sab = defaultdict(nx.DiGraph)
    with open(mrrel_rrf_path, 'rt') as mrrel:
        reader = csv.reader(mrrel, delimiter='|')
        for row in tqdm.tqdm(reader, desc="Reading MRREL.RRF", unit='rows', total=80256532):
            # docs: https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.related_concepts_file_mrrel_rrf
            cui1, aui1, stype1, rel, cui2, aui2, stype2, rela, rui, srui, sab, sl, rg, dir_, suppress, cvf, _ = row
            if suppress != "N":
                continue
            # parent -> child and broader -> narrower relations encode the hierarchy
            if cui1 != cui2 and (rel == "CHD" or rel == "RN"):
                hier = hiers_by_sab[sab]  # SAB is the abbreviated name of the source vocabulary
                if hier.has_edge(cui1, cui2):
                    metadata = hier.edges[cui1, cui2]["metadata"]
                else:
                    hier.add_edge(cui1, cui2)
                    metadata = list()
                    hier.edges[cui1, cui2]["metadata"] = metadata
                metadata.append((sab, rel, rela, rui))

    for sab, hier in hiers_by_sab.items():
        if sab == 'MTH':  # The Metathesaurus added relations do not form an ontology.
            continue
        subgraph_nodes = set()
        for cui in target_cuis:
            if cui in hier.nodes:
                subgraph_nodes.add(cui)
                subgraph_nodes.update(nx.ancestors(hier, cui))
        # Only include targeted concepts and their ancestors.
        hiers_by_sab[sab] = nx.DiGraph(hier.subgraph(subgraph_nodes))

    hiers_by_sab = {k: v for k, v in hiers_by_sab.items() if any(v.nodes)}  # remove trees with no targets in them
    roots = {k: {n for n in v.nodes if v.in_degree[n] == 0 and v.out_degree[n] >= 1} for k, v in hiers_by_sab.items()}
    roots = {k: next(iter(v)) for k, v in roots.items() if len(v) == 1}
    roots |= {'MSH': 'C1135584', 'GO': 'C1138831', 'HPO': 'C4020635', 'MEDCIN': 'C2182611', 'NCI': 'C1140168'}

    # Cover the target concepts while using as few different sources as possible. (Using the method by V. Chvatal, 1979)
    comps_by_root = {r: nx.bfs_tree(hiers_by_sab[s], r) for s, r in roots.items()}
    leaf_count = {r: len(set(comps_by_root[r].nodes) & target_cuis) for r in roots.values()}
    comps_by_root_copy = dict(comps_by_root.items())
    found_trees = []
    found_cuis = set()
    for _ in range(len(comps_by_root_copy)):
        scores = {s: 1 / len(set(tree.nodes) & target_cuis - found_cuis) for s, tree in
                  comps_by_root_copy.items()
                  if len(set(tree.nodes) & target_cuis - found_cuis) > 0}
        if not any(scores):
            break
        best_source, _ = min(scores.items(), key=lambda x: x[1])
        best_tree = comps_by_root_copy[best_source]
        subtree_nodes = set()
        for cui in target_cuis - found_cuis:
            if cui in best_tree.nodes:
                subtree_nodes.add(cui)
                subtree_nodes.update(nx.ancestors(best_tree, cui))
        found_trees.append(
            (best_source, best_tree.subgraph(subtree_nodes)))
        found_cuis.update(comps_by_root_copy[best_source])
        del comps_by_root_copy[best_source]
    hier: nx.DiGraph = nx.compose_all([tree for src, tree in found_trees])

    for n in list(hier.nodes):
        if hier.in_degree[n] == 0:
            hier.add_edge('root', n)
    assert nx.number_weakly_connected_components(hier) == 1

    # Theese "trees" are not quite trees yet. Not even a DAG.
    # Some loops have to be removed first.
    leaves = {n for n in hier.nodes if hier.out_degree[n] == 0}
    depth = dict()
    not_found = set(hier.nodes)
    for i, layer in enumerate(bfs_layers(hier, ['root'])):
        found = 0
        for n in layer:
            not_found.discard(n)
            if n not in depth:
                found += 1
                depth[n] = i
        if found == 0:
            break
    hier.remove_nodes_from(not_found)
    for x, y in set(hier.edges):
        if depth[x] >= depth[y]:
            hier.remove_edge(x, y)

    assert nx.is_directed_acyclic_graph(hier)
    assert nx.number_weakly_connected_components(hier) == 1

    # Metadata were lost in the set covering process. Add them back in.
    for hier_orig in hiers_by_sab.values():
        for cui1, cui2 in hier_orig.edges:
            if hier.has_edge(cui1, cui2):
                edge_attrs = hier.edges[cui1, cui2]
                orig_metadata = hier_orig.edges[cui1, cui2]["metadata"]
                if "metadata" in edge_attrs:
                    edge_attrs["metadata"].extend(orig_metadata)
                else:
                    edge_attrs["metadata"] = orig_metadata

    return hier


def umls_relation_subset(target_cuis: Container[str], mrrel_rrf_path):
    rels = nx.DiGraph()
    with open(mrrel_rrf_path, 'rt') as mrrel:
        reader = csv.reader(mrrel, delimiter='|')
        for row in tqdm.tqdm(reader, desc="Reading MRREL.RRF", unit='rows', total=80256532):
            # docs: https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.related_concepts_file_mrrel_rrf
            cui1, aui1, stype1, rel, cui2, aui2, stype2, rela, rui, srui, sab, sl, rg, dir_, suppress, cvf, _ = row
            if suppress != "N":
                continue
            if rel == 'SIB':  # Sibling relations hold little relevant information here.
                continue
            if cui1 != cui2 and cui1 in target_cuis and cui2 in target_cuis:
                if rels.has_edge(cui1, cui2):
                    metadata = rels.edges[cui1, cui2]["metadata"]
                else:
                    rels.add_edge(cui1, cui2)
                    metadata = list()
                    rels.edges[cui1, cui2]["metadata"] = metadata
                metadata.append((sab, rel, rela, rui))
    return rels


def get_hgnc2ensp(target_ensp_set: Container[str],
                  hgnc2ensp_tsv_path, hgnc2ensp_oct2014_tsv_path):
    # STRING v11.5 uses Ensembl protein IDs from the outdated
    # Ensembl version 77, GRCh38, released Oct. 2014, updated Aug. 2014.
    # Some of these IDs have been removed from subsequent Ensembl releases.
    # Here both the latest Ensembl and the old Ensembl are used in order to
    # include both the deprecated proteins and the more recent protein-gene
    # associations in the results.

    query_xml = \
        '<?xml version="1.0" encoding="UTF-8"?>' \
        '<!DOCTYPE Query>' \
        '<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" datasetConfigVersion="0.6">' \
        '<Dataset name="hsapiens_gene_ensembl" interface="default">' \
        '<Attribute name="hgnc_id"/>' \
        '<Attribute name="ensembl_peptide_id"/>' \
        '<Attribute name="hgnc_symbol"/>' \
        '</Dataset>' \
        '</Query>'
    if not os.path.exists(hgnc2ensp_tsv_path):
        url = f'https://www.ensembl.org/biomart/martservice?query=' + urllib.parse.quote(query_xml)
        urllib.request.urlretrieve(url, hgnc2ensp_tsv_path)

    hgnc2ensp_entries: set[Tuple[int, int]] = set()
    ensp2sym_entries: set[Tuple[int, str]] = set()
    with open(hgnc2ensp_tsv_path, 'rt') as tsv:
        for row in csv.DictReader(tsv, delimiter='\t'):
            if row['HGNC ID'] and row['Protein stable ID'].startswith('ENSP'):
                hgnc_id = int(row['HGNC ID'][5:])
                ensembl_peptide_id = int(row['Protein stable ID'][4:])
                hgnc2ensp_entries.add((hgnc_id, ensembl_peptide_id))
            if row['HGNC symbol'] and row['Protein stable ID'].startswith('ENSP'):
                ensp2sym_entries.add((int(row['Protein stable ID'][4:]), row['HGNC symbol']))

    if not os.path.exists(hgnc2ensp_oct2014_tsv_path):
        url = f'https://oct2014.archive.ensembl.org/biomart/martservice?query=' \
              + urllib.parse.quote(query_xml)
        urllib.request.urlretrieve(url, hgnc2ensp_oct2014_tsv_path)

    with open(hgnc2ensp_oct2014_tsv_path, "rt") as tsv:
        for row in csv.DictReader(tsv, delimiter='\t'):
            if row['HGNC ID(s)'] and row['Ensembl Protein ID'].startswith('ENSP'):
                hgnc_id = int(row['HGNC ID(s)'][5:])
                ensembl_peptide_id = int(row['Ensembl Protein ID'][4:])
                hgnc2ensp_entries.add((hgnc_id, ensembl_peptide_id))
            if row['HGNC symbol'] and row['Ensembl Protein ID'].startswith('ENSP'):
                ensp2sym_entries.add((int(row['Ensembl Protein ID'][4:]), row['HGNC symbol']))

    hgnc2ensp = defaultdict(set)
    for hgnc, ensp in hgnc2ensp_entries:
        ensp = f"9606.ENSP{ensp:011}"
        if ensp in target_ensp_set:
            hgnc2ensp[hgnc].add(ensp)

    ensp2sym = defaultdict(set)
    for ensp, sym in ensp2sym_entries:
        ensp = f"9606.ENSP{ensp:011}"
        if ensp in target_ensp_set:
            ensp2sym[ensp].add(sym)
    return hgnc2ensp, ensp2sym


def matrix_permute(mat: sp.spmatrix, perm_row: np.ndarray, perm_col: np.ndarray = None) -> sp.csr_matrix:
    if perm_col is None:
        perm_col = perm_row
    m, n = mat.shape
    perm_row_mat = sp.coo_matrix((np.ones(n, dtype=mat.dtype), (np.arange(n, dtype=perm_row.dtype), perm_row)))
    perm_col_mat = sp.coo_matrix((np.ones(m, dtype=mat.dtype), (perm_col, np.arange(m, dtype=perm_col.dtype))))
    return perm_row_mat.tocsr() @ mat @ perm_col_mat.tocsr()


def reverse_cuthill_mckee(adjacency: sp.csr_matrix, node_labels: Sequence, symmetric_mode=True):
    """Reverse Cuthill McKee order the labeled graph."""
    permutation = sp.csgraph.reverse_cuthill_mckee(adjacency, symmetric_mode)
    new_adj = matrix_permute(adjacency, permutation, permutation)
    new_labels: list = [node_labels[p] for p in permutation]
    return new_adj, new_labels


def trim_matrix(mat: sp.csr_matrix, square=True):
    """Remove rows and columns with zeros from the end of the matrix."""
    mat.sort_indices()
    rows = np.max(np.where(np.diff(mat.indptr) > 0)[0]) + 1
    cols = np.max(mat.indices) + 1
    if square:
        rows = cols = max(rows, cols)
    mat._shape = (rows, cols)
    mat.indptr = mat.indptr[:rows + 1]


def ppi_graph_to_matrix(protein_links_path, clusters_tree_path, clusters_proteins_path):
    ppi_raw = load_protein_graph(protein_links_path)
    ppi_raw.data = ppi_raw.data.astype(np.float64) / 1000.0
    protein_id_range = range(np.max(np.concatenate((ppi_raw.row, ppi_raw.col))) + 1)
    ppi_raw = ppi_raw.tocsr()
    adj_cmk, node2ensp = reverse_cuthill_mckee(ppi_raw, protein_id_range, symmetric_mode=True)
    trim_matrix(adj_cmk, square=True)
    node2ensp = node2ensp[:adj_cmk.shape[0]]
    node2ensp = [f"9606.ENSP{id:011}" for id in node2ensp]

    if not os.path.exists('protein_feat1024.npz'):
        # TODO: proper caching
        po = protein_ontology(clusters_tree_path, clusters_proteins_path)
        proteins = sorted({n for n in po if po.out_degree[n] == 0})
        feat = ontology_feature_encoding(po, proteins, dim=1024)
        np.savez_compressed("protein_feat1024.npz", feat=feat, proteins=proteins)

    npz = np.load("protein_feat1024.npz")
    feat, proteins = npz["feat"], npz["proteins"]
    protein2feat = dict(zip(proteins, feat))
    node_feat = np.vstack([protein2feat[c] for c in node2ensp])

    return adj_cmk.tocoo(), node_feat, node2ensp


def create_disgenet_graph(disgenet_db_path, score_threshold=0.0):
    disgenet_hgnc2cui = nx.DiGraph()
    with sqlite3.connect(disgenet_db_path) as conn:
        sqlcur = conn.cursor()
        for geneId, diseaseId, score in sqlcur.execute(
                "select geneId, diseaseId, score from geneDiseaseNetwork "
                "JOIN diseaseAttributes USING (diseaseNID) JOIN geneAttributes USING (geneNID)"
                "WHERE score >= ?;", (score_threshold,)
        ):
            if not disgenet_hgnc2cui.has_edge(geneId, diseaseId):
                disgenet_hgnc2cui.add_edge(geneId, diseaseId)
                disgenet_hgnc2cui.edges[geneId, diseaseId]["score"] = score
            else:
                assert disgenet_hgnc2cui.edges[geneId, diseaseId]["score"] == score

    return disgenet_hgnc2cui


def disease_graph_to_matrix(target_cuis: Iterable[str], mrrel_rrf_path):
    dg_hier = umls_ontology_subset(target_cuis, mrrel_rrf_path)

    target_cuis = set(dg_hier.nodes)
    dg_rels = umls_relation_subset(target_cuis, mrrel_rrf_path)

    nodelist = list(dg_rels.nodes)
    adj: sp.csr_matrix = nx.to_scipy_sparse_array(dg_rels, nodelist=nodelist, weight=None, format='csr')
    adj_cmk, node2cui = reverse_cuthill_mckee(adj, nodelist, symmetric_mode=True)

    dg_feat = ontology_feature_encoding(dg_hier, node2cui, dim=1024)
    cui2feat = dict(zip(node2cui, dg_feat))

    node_feat = np.vstack([cui2feat[c] for c in node2cui])
    return adj_cmk.tocoo(), node_feat, node2cui


def resnik_similarity(concepts: Sequence, ic: Mapping[Any, float], hypernyms: Mapping[Any, set]):
    """
    P. Resnik: Semantic Similarity in a Taxonomy: An Information-Based Measure
    and its Application to Problems of Ambiguity in Natural Language (1999) https://doi.org/10.1613/jair.514
    """
    progress = tqdm.tqdm(total=len(concepts) * (len(concepts) + 1) // 2,
                         desc="Computing Resnik similarity", unit="concept pairs")
    similarity = np.zeros((len(concepts), len(concepts)), dtype=np.float64)
    for x in range(len(concepts)):
        for y in range(x, len(concepts)):
            common_ancestors = hypernyms[concepts[x]] & hypernyms[concepts[y]]
            vals = np.empty((len(common_ancestors),), dtype=np.float64)
            for i, n in enumerate(common_ancestors):  # looping is faster than np.fromiter()
                vals[i] = ic[n]
            similarity[x, y] = np.maximum.reduce(vals)  # much faster than np.max() or the Python max()
        progress.update(len(concepts) - x)
    similarity = similarity + np.tril(similarity.T, k=-1)  # Make it symmetric. Take care not to alter the diagonal.
    return similarity


def lin_similarity(concepts: Sequence, resnik: np.ndarray, ic: Mapping[Any, float]):
    """
    This is advantageous over the Resnik similarity because it's constrained to the interval [0, 1]
    and it's invariant to IC scaling.

    D. Lin: An information-theoretic definition of similarity (1998) https://api.semanticscholar.org/CorpusID:5659557
    """
    progress = tqdm.tqdm(total=len(concepts) * (len(concepts) + 1) // 2,
                         desc="Computing Lin similarity", unit="concept pairs")
    similarity = np.zeros((len(concepts), len(concepts)), dtype=np.float64)
    for x in range(len(concepts)):
        for y in range(x, len(concepts)):
            denominator = ic[concepts[x]] + ic[concepts[y]]
            similarity[x, y] = 2 * resnik[x, y] / denominator if denominator != 0 else 0
        progress.update(len(concepts) - x)
    similarity = similarity + np.tril(similarity.T, k=-1)  # Make it symmetric. Take care not to alter the diagonal.
    return similarity


def ontology_feature_encoding(ontology: nx.DiGraph, target_concepts: Sequence, dim=64, eps=1e-9,
                              use_sanchez_ic=False, use_lin_similarity=False, error_measurement=False):
    """
    The ontology is a DAG where the nodes are concepts and edges denote "inverse isa" relations.
    For each target concept in the ontology assign a vector of dim dimensions.
    The dot product of two of these vectors will approximately equal to the similarity between the concepts.
    """

    leaves = {n for n in ontology if ontology.out_degree[n] == 0}
    hypernyms = {n: nx.ancestors(ontology, n) for n in ontology.nodes}
    for n in ontology.nodes:
        hypernyms[n].add(n)
    hyponyms = {n: nx.descendants(ontology, n) for n in ontology.nodes}
    for n in ontology.nodes:
        hyponyms[n].add(n)

    # intrinsic information content
    if use_sanchez_ic:
        # D. Sánchez et al.: Ontology-based information content computation (2011)
        # https://doi.org/10.1016/j.knosys.2010.10.001
        ic = {n: np.abs(np.log2(((len(hyponyms[n] & leaves) / len(hypernyms[n])) + 1) / (len(leaves) + 1)))
              for n in ontology.nodes}
    else:
        ic = {n: np.abs(np.log2(len(hyponyms[n] & leaves) / len(ontology.nodes)))
              for n in ontology.nodes}

    similarity = resnik_similarity(target_concepts, ic, hypernyms)

    if use_lin_similarity:
        similarity = lin_similarity(target_concepts, similarity, ic)

    print("Cholesky decomposition")
    try:
        ch_tril = np.linalg.cholesky(similarity)
    except np.linalg.LinAlgError as err:
        assert "Matrix is not positive definite" in err.args
        # Here's a little hack to make the matrix positive definite.
        # Right now some eigenvalues are <= 0, but we want all of them > 0.
        # Find the top k largest magnitude ("LM") eigenvalues.
        # The first one of these that's negative is the minimum.
        # Iteratively increase k because it gets harder to compute.
        # We're hoping to catch the negative eigenvalue before having to reach large values of k.
        min_eigval = None
        for k in range(10, similarity.shape[0] - 1, 30):
            eigvals = sp.linalg.eigsh(similarity, k=k, which="LM", return_eigenvectors=False)
            if np.any(eigvals < 0):
                min_eigval = np.min(eigvals)
                break
            else:
                print(f"debug: The {k} largest magnitude eigenvalues didn't contain negatives!")
        assert min_eigval is not None
        print(f"{min_eigval=}")
        # Subtract the minimum to make the eigenvalues >= 0. Add a small epsilon to make them > 0.
        # This is okay to do because we don't care about the self-similarities of concepts.
        sim_pd = similarity - np.eye(*similarity.shape) * (min_eigval - eps)
        ch_tril = np.linalg.cholesky(sim_pd)

    print("SVD")
    # algorithm="arpack" is slower than "randomized" but it's more accurate
    svd = TruncatedSVD(n_components=dim, algorithm="arpack").fit_transform(ch_tril)

    if error_measurement:
        l2_error = 0.0
        l2_sim = 0.0
        l2_approx = 0.0
        greatly_differing_items = 0
        progress = tqdm.tqdm(total=len(target_concepts) * (len(target_concepts) + 1) // 2,
                             desc="Error measurement", unit="concept pairs")
        for x in range(len(target_concepts)):
            sim = similarity[x, x:]
            approx = svd[x] @ svd[x:].T
            approx[0] = sim[0]  # disregard the diagonal, which was altered on purpose
            l2_error += np.linalg.norm(sim - approx) ** 2
            l2_sim += np.linalg.norm(sim) ** 2
            l2_approx += np.linalg.norm(approx) ** 2
            greatly_differing_items += np.count_nonzero(np.round(sim) != np.round(approx))
            progress.update(len(target_concepts) - x)
        l2_error *= 2.0  # account for the lower triangle of the matrix
        l2_sim *= 2.0
        l2_approx *= 2.0
        greatly_differing_items *= 2
        l2_error = np.sqrt(l2_error)
        l2_sim = np.sqrt(l2_sim)
        l2_approx = np.sqrt(l2_approx)
        print(f"{l2_error=} {l2_sim=} {l2_approx=} {greatly_differing_items=}")
    return svd


def get_cui2name(mrconso_rrf_path='MRCONSO.RRF', disgenet_db_path='disgenet_2020.db'):
    cui2name = dict()
    with sqlite3.connect(disgenet_db_path) as conn:
        sqlcur = conn.cursor()
        disgenet_rows = list(sqlcur.execute(
            "SELECT DISTINCT diseaseID, diseaseName from diseaseAttributes;"
        ))
        for cui, name in disgenet_rows:
            cui2name[cui] = name

    mrconso_entries = defaultdict(list)
    with open(mrconso_rrf_path, 'rt') as mrconso:
        reader = csv.reader(mrconso, delimiter="|")
        for row in tqdm.tqdm(reader, total=14608809):
            # docs: https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr
            cui, lat, ts, lui, stt, sui, ispref, aui, saui, scui, sdui, sab, tty, code, str_, srl, suppress, cvf, _ = row
            if cui not in cui2name:
                mrconso_entries[cui].append((str_, tty, lat, ispref, suppress))

    for cui, entries in mrconso_entries.items():
        pns = [str_ for str_, tty, lat, ispref, suppress in entries
               if lat == "ENG" and tty == 'PN']
        if any(pns):
            cui2name[cui] = pns[0]
            continue
        prefs = [str_ for str_, tty, lat, ispref, suppress in entries
                 if lat == "ENG" and ispref == 'Y' and suppress == 'N']
        if any(prefs):
            cui2name[cui] = prefs[0]
            continue
        nonsup = [str_ for str_, tty, lat, ispref, suppress in entries
                  if lat == "ENG" and suppress == 'N']
        if any(nonsup):
            cui2name[cui] = nonsup[0]
            continue
        engsup = [str_ for str_, tty, lat, ispref, suppress in entries
                  if lat == "ENG"]
        if any(nonsup):
            cui2name[cui] = engsup[0]
            continue
        if any(entries):
            cui2name[cui] = entries[0]

    return cui2name


def create_train_dataset(args):
    ppi_adj_cmk, ppi_feat, prot2ensp = ppi_graph_to_matrix(
        args.string_protein_links_path, args.string_clusters_tree_path, args.string_clusters_proteins_path
    )

    hgnc2ensp, ensp2sym = get_hgnc2ensp(set(prot2ensp), args.hgnc2ensp_tsv_path, args.hgnc2ensp_oct2014_tsv_path)
    disgenet_hgnc2cui = create_disgenet_graph(args.disgenet_db_path)

    disgenet_ensp2cui = nx.DiGraph()
    for g, d in disgenet_hgnc2cui.edges:
        if g in hgnc2ensp:
            score = disgenet_hgnc2cui.edges[g, d]["score"]
            for ensp in hgnc2ensp[g]:
                if not disgenet_ensp2cui.has_edge(ensp, d):
                    disgenet_ensp2cui.add_edge(ensp, d)
                    disgenet_ensp2cui.edges[ensp, d]["score"] = score
                else:
                    prev_score = disgenet_ensp2cui.edges[ensp, d]["score"]
                    disgenet_ensp2cui.edges[ensp, d]["score"] = max(prev_score, score)

    disease_scores = defaultdict(list)
    for g, d in disgenet_ensp2cui.edges:
        disease_scores[d].append(disgenet_ensp2cui.edges[g, d]["score"])

    # target_cuis = {d for d, sc in disease_scores.items()
    #                if sum(1 for s in sc if s >= 0.3) >= 2 or sum(1 for s in sc if s >= 0.1) >= 10}

    target_cuis = {d for d, sc in disease_scores.items() if sum(sc) >= 0.6}

    dis_adj_cmk, dis_feat, dis2cui = disease_graph_to_matrix(target_cuis, args.umls_mrrel_rrf_path)

    ensp2node = dict((ensp, node) for node, ensp in enumerate(prot2ensp))
    cui2node = dict((cui, node) for node, cui in enumerate(dis2cui))
    disgenet_proteins, disgenet_diseases, disgenet_scores = [], [], []
    for ensp, cui in disgenet_ensp2cui.edges:
        if cui in cui2node and ensp in ensp2node:
            disgenet_proteins.append(ensp2node[ensp])
            disgenet_diseases.append(cui2node[cui])
            disgenet_scores.append(disgenet_ensp2cui.edges[ensp, cui]["score"])

    prot2name = [sorted(ensp2sym[ensp])[0] if ensp in ensp2sym else ensp for ensp in prot2ensp]
    cui2name = get_cui2name(args.umls_mrconso_rrf_path, args.disgenet_db_path)
    dis2name = [cui2name[cui] for cui in dis2cui]

    data = HeteroData()
    data['protein'].x = torch.from_numpy(ppi_feat).float()
    data['protein'].label = np.array(prot2ensp)
    data['protein'].name = np.array(prot2name)

    data['disease'].x = torch.from_numpy(dis_feat).float()
    data['disease'].label = np.array(dis2cui)
    data['disease'].name = np.array(dis2name)

    data['protein', 'associated_with', 'protein'].edge_index = torch.from_numpy(
        np.vstack((ppi_adj_cmk.row, ppi_adj_cmk.col))).long()
    data['protein', 'associated_with', 'protein'].edge_attr = torch.from_numpy(
        np.expand_dims(ppi_adj_cmk.data, axis=1)).float()

    data['disease', 'associated_with', 'disease'].edge_index = torch.from_numpy(
        np.vstack((dis_adj_cmk.row, dis_adj_cmk.col))).long()
    data['disease', 'associated_with', 'disease'].edge_attr = torch.ones((dis_adj_cmk.nnz, 1), dtype=torch.float32)

    data['protein', 'associated_with', 'disease'].edge_index = torch.from_numpy(
        np.vstack((disgenet_proteins, disgenet_diseases))).long()
    data['protein', 'associated_with', 'disease'].edge_attr = torch.from_numpy(
        np.expand_dims(disgenet_scores, axis=1)).float()

    data['disease', 'associated_with', 'protein'].edge_index = torch.from_numpy(
        np.vstack((disgenet_diseases, disgenet_proteins))).long()
    data['disease', 'associated_with', 'protein'].edge_attr = torch.from_numpy(
        np.expand_dims(disgenet_scores, axis=1)).float()

    torch.save(data, args.output_path)


def get_phrase2pheno(input_path='gwasATLAS_v20191115.txt.gz'):
    phrase2gwas = defaultdict(set)
    cat2gwas = defaultdict(set)
    with gzip.open(input_path, 'rt') as gwas_atlas_phenotypes:
        reader = csv.reader(gwas_atlas_phenotypes, delimiter='\t')
        header = next(reader)
        id_col, chapter_level_col, subchapter_level_col, trait_col, uniq_trait_col = \
            header.index('id'), header.index('ChapterLevel'), header.index('SubchapterLevel'), \
            header.index('Trait'), header.index('uniqTrait')
        for row in reader:
            id_, chapter_level, subchapter_level, trait, uniq_trait = \
                (int(row[id_col]), row[chapter_level_col], row[subchapter_level_col],
                 row[trait_col], row[uniq_trait_col])
            cat2gwas[chapter_level].add(id_)
            cat2gwas[subchapter_level].add(id_)
            if '::' in trait:
                components = trait.split('::')
                assert len(components) == 3
                cat2gwas[components[0]].add(id_)
                cat2gwas[components[1]].add(id_)
                phrase2gwas[components[2]].add(id_)
            else:
                phrase2gwas[trait].add(id_)
            assert '::' not in uniq_trait
            phrase2gwas[uniq_trait].add(id_)
    del cat2gwas['']
    return phrase2gwas, cat2gwas

def _match_to_umls(phrases: Sequence[str], threshold=0.8, mrconso_rrf_path='MRCONSO.RRF'):
    def normalize(phrase: str):
        phrase = phrase.lower()
        phrase = phrase.replace('\'', '').replace('ï', 'i')
        phrase = re.sub(r'[\-"()\[\]/\\;?_#&,*.]', ' ', phrase)
        return re.sub(' +', ' ', phrase).lstrip().rstrip()

    def shingles(phrase: str, size=2):
        padded = ' ' * (size - 1) + phrase + ' ' * (size - 1)
        for i in range(len(phrase) + (size - 1)):
            yield padded[i:i + size]

    occurrences = {ph: Counter(shingles(normalize(ph), 2)) for ph in phrases}
    all_shingles = sorted(set.union(*(set(occ.keys()) for occ in occurrences.values())))
    occurrence_vectors_dense = np.array([[occurrences[ph][sh] for ph in phrases] for sh in all_shingles],
                                        dtype=np.float64)
    occurrence_vectors_dense = occurrence_vectors_dense / np.linalg.norm(occurrence_vectors_dense, axis=0)
    occurrence_vectors: sp.csr_matrix = sp.csr_matrix(occurrence_vectors_dense)

    matches = []
    with open(mrconso_rrf_path, 'rt') as mrconso:
        found_luis = set()
        reader = csv.reader(mrconso, delimiter="|")
        for i, row in enumerate(tqdm.tqdm(reader, total=14_608_809)):
            # docs: https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr
            cui, lat, ts, lui, stt, sui, ispref, aui, saui, scui, sdui, sab, tty, code, str_, srl, suppress, cvf, _ = row
            if lat != "ENG":
                continue
            lui_int = int(lui[1:])
            if lui_int in found_luis:
                continue
            found_luis.add(lui_int)
            occ = Counter(shingles(normalize(str_), 2))
            vec = np.array([occ[sh] for sh in all_shingles], dtype=np.float64)
            vec = vec / np.linalg.norm(vec)
            sim: np.ndarray = vec @ occurrence_vectors
            for phrase_idx in np.where(sim >= threshold)[0]:
                score = sim[phrase_idx]
                matches.append((cui, lui, phrase_idx, score, str_))
    return matches


def match_to_umls(input_path='gwasATLAS_v20191115.txt.gz', mrconso_rrf_path='MRCONSO.RRF'
                  ) -> dict[int, dict[str, float]]:
    trait2gwas, cat2gwas = get_phrase2pheno(input_path)

    gwas_ids = set.union(*(trait2gwas.values()))
    all_phrases = sorted(set(trait2gwas.keys() | cat2gwas.keys()))
    all_matches = _match_to_umls(all_phrases, threshold=0.8, mrconso_rrf_path=mrconso_rrf_path)
    all_matches = sorted(all_matches, key=lambda x: x[3], reverse=True)

    phrase2match = defaultdict(dict)
    for cui, lui, phrase_idx, score, str_ in all_matches:
        matches = phrase2match[all_phrases[phrase_idx]]
        if cui not in phrase2match:
            matches[cui] = score

    gwas2trait = defaultdict(list)
    for trait, ids in trait2gwas.items():
        for id_ in ids:
            gwas2trait[id_].append(trait)
    for traits in gwas2trait.values():
        traits.sort(key=lambda ph: len(trait2gwas[ph]))

    gwas2cuis = defaultdict(dict)
    for id_, traits in gwas2trait.items():
        for trait in traits:
            if trait in phrase2match:
                for cui, score in phrase2match[trait].items():
                    gwas2cuis[id_][cui] = max(score, gwas2cuis[id_].get(cui, 0))

    for id_ in list(gwas2trait.keys()):
        if len(gwas2cuis[id_]) == 0:
            del gwas2cuis[id_]

    gwas2cats = defaultdict(list)
    for cat, ids in cat2gwas.items():
        for id_ in ids:
            gwas2cats[id_].append(cat)
    gwas2cat = {id_: sorted(gwas2cats[id_], key=lambda c: len(cat2gwas[c]))[0] for id_ in gwas2cats.keys()}

    for id_ in gwas_ids:
        if id_ not in gwas2cuis:
            gwas2cuis[id_] = phrase2match[gwas2cat[id_]]

    return {id_: (gwas2trait[id_], cuis) for id_, cuis in gwas2cuis.items()}


def get_ensg2ensp(ensg2ensp_tsv_path='ensg2ensp.tsv',
                  ensg2ensp_oct2014_tsv_path='ensg2ensp_oct2014.tsv',
                  ensg2ensp_grch37_tsv_path='ensg2ensp_grch37.tsv') -> Mapping[int, set[str]]:
    query_xml = \
        '<?xml version="1.0" encoding="UTF-8"?>' \
        '<!DOCTYPE Query>' \
        '<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" datasetConfigVersion="0.6">' \
        '<Dataset name="hsapiens_gene_ensembl" interface="default">' \
        '<Attribute name="ensembl_gene_id"/>' \
        '<Attribute name="ensembl_peptide_id"/>' \
        '</Dataset>' \
        '</Query>'

    # Get the latest, most up-to-date infos.
    if not os.path.exists(ensg2ensp_tsv_path):
        url = f'https://www.ensembl.org/biomart/martservice?query=' + urllib.parse.quote(query_xml)
        urllib.request.urlretrieve(url, ensg2ensp_tsv_path)

    # Get oct2014 to match STRING v11.5.
    if not os.path.exists(ensg2ensp_oct2014_tsv_path):
        url = f'https://oct2014.archive.ensembl.org/biomart/martservice?query=' + urllib.parse.quote(query_xml)
        urllib.request.urlretrieve(url, ensg2ensp_oct2014_tsv_path)

    # Get grch37 to match gwasAtlas.
    if not os.path.exists(ensg2ensp_grch37_tsv_path):
        url = f'https://grch37.ensembl.org/biomart/martservice?query=' + urllib.parse.quote(query_xml)
        urllib.request.urlretrieve(url, ensg2ensp_grch37_tsv_path)

    ensg2ensp_entries: set[Tuple[str, str]] = set()
    with open(ensg2ensp_tsv_path, 'rt') as tsv:
        for row in csv.DictReader(tsv, delimiter='\t'):
            if row['Gene stable ID'] and row['Protein stable ID'].startswith('ENSP'):
                ensembl_gene_id = row['Gene stable ID']
                ensembl_peptide_id = row['Protein stable ID']
                ensg2ensp_entries.add((ensembl_gene_id, ensembl_peptide_id))

    with open(ensg2ensp_oct2014_tsv_path, 'rt') as tsv:
        for row in csv.DictReader(tsv, delimiter='\t'):
            if row['Ensembl Gene ID'] and row['Ensembl Protein ID'].startswith('ENSP'):
                ensembl_gene_id = row['Ensembl Gene ID']
                ensembl_peptide_id = row['Ensembl Protein ID']
                ensg2ensp_entries.add((ensembl_gene_id, ensembl_peptide_id))

    with open(ensg2ensp_grch37_tsv_path, 'rt') as tsv:
        for row in csv.DictReader(tsv, delimiter='\t'):
            if row['Gene stable ID'] and row['Protein stable ID'].startswith('ENSP'):
                ensembl_gene_id = row['Gene stable ID']
                ensembl_peptide_id = row['Protein stable ID']
                ensg2ensp_entries.add((ensembl_gene_id, ensembl_peptide_id))

    ensg2ensp = defaultdict(set)
    for ensg, ensp in ensg2ensp_entries:
        ensg2ensp[ensg].add('9606.' + ensp)
    return ensg2ensp


def get_gwas2prot(ensg2ensp, gwasatlas_magma_path='gwasATLAS_v20191115_magma_P.txt.gz'):
    genes = []
    values = []
    with gzip.open(gwasatlas_magma_path, 'rt') as protein_links:
        reader = csv.reader(protein_links, delimiter='\t')
        header = next(reader)
        ids = [int(id_) for id_ in header[1:]]
        for row in tqdm.tqdm(reader, total=20187, unit='rows'):
            genes.append(row[0])
            values.append([float(x) if x != 'NA' else float('nan') for x in row[1:]])
    p_raw = np.array(values)

    ensps = sorted(list(itertools.chain.from_iterable(ensg2ensp.values())))
    ensp2num = {ensp: i for i, ensp in enumerate(ensps)}

    gwas2prot = defaultdict(list)
    for col, id_ in tqdm.tqdm(enumerate(ids), total=len(ids)):
        for i, p in enumerate(p_raw[:, col]):
            if not np.isnan(p):
                ensg = genes[i]
                if ensg in ensg2ensp:
                    for ensp in ensg2ensp[ensg]:
                        gwas2prot[id_].append((ensp2num[ensp], p))
        gwas2prot[id_].sort(key=lambda x: x[1], reverse=False)
        gwas2prot[id_] = np.array(gwas2prot[id_])
    return ensps, gwas2prot


def create_gwasatlas_dataset(args):
    ppi_raw = load_protein_graph(args.string_protein_links_path)
    string_proteins = set(ppi_raw.row) | set(ppi_raw.col)
    string_proteins = {f"9606.ENSP{id:011}" for id in string_proteins}
    ensg2ensp = get_ensg2ensp(args.ensg2ensp_tsv_path, args.ensg2ensp_oct2014_tsv_path, args.ensg2ensp_grch37_tsv_path)
    ensg2ensp = {ensg: ensps & string_proteins for ensg, ensps in ensg2ensp.items() if any(ensps & string_proteins)}
    gwas2prot = get_gwas2prot(ensg2ensp, args.gwasatlas_magma_path)
    gwas2prot: Tuple[list[str], dict[int, np.ndarray]]
    with open(args.gwas2prot_picke_path, "wb") as f:
        pickle.dump(gwas2prot, f)

    gwas2cuis = match_to_umls(args.gwasatlas_path, args.umls_mrconso_rrf_path)
    gwas2cuis: dict[int, Tuple[list[int], dict[str, float]]]
    with open(args.gwas2cuis_picke_path, "wb") as f:
        pickle.dump(gwas2cuis, f)


def main():
    # TODO: add help messages
    # TODO: add quiet arg: parser.add_argument('-q', '--quiet', action="store_true")
    # TODO: add arguments for ontology_feature_encoding

    parser = argparse.ArgumentParser(description='Create the dataset.')
    subparsers = parser.add_subparsers(title='Subcommands', dest='subcommand')

    train = subparsers.add_parser('train', help='Create the training dataset')

    train.add_argument('--output-path', default='protein_disease_hetero_graph.pt',
                       required=False, type=argparse.FileType('wb'))
    train.add_argument('--string-protein-links-path',
                       default='9606.protein.links.full.v11.5.txt.gz',
                       required=False, type=pathlib.PurePath)
    train.add_argument('--string-clusters-tree-path',
                       default='9606.clusters.tree.v11.5.txt.gz',
                       required=False, type=pathlib.PurePath)
    train.add_argument('--string-clusters-proteins-path',
                       default='9606.clusters.proteins.v11.5.txt.gz',
                       required=False, type=pathlib.PurePath)
    train.add_argument('--disgenet-db-path', default='disgenet_2020.db',
                       required=False, type=pathlib.PurePath)
    train.add_argument('--umls-mrrel-rrf-path', default='MRREL.RRF',
                       required=False, type=pathlib.PurePath)
    train.add_argument('--umls-mrconso-rrf-path', default='MRCONSO.RRF',
                       required=False, type=pathlib.PurePath)
    train.add_argument('--hgnc2ensp-tsv-path', default='hgnc2ensp.tsv',
                       required=False, type=pathlib.PurePath)
    train.add_argument('--hgnc2ensp-oct2014-tsv-path', default='hgnc2ensp_oct2014.tsv',
                       required=False, type=pathlib.PurePath)

    ga = subparsers.add_parser('gwasatlas', help='Create the gwasatlas dataset')

    ga.add_argument('--gwas2cuis-picke-path', default='gwas2cuis.pickle', help="output",
                    required=False, type=pathlib.PurePath)
    ga.add_argument('--gwas2prot-picke-path', default='gwas2prot.pickle', help="output",
                    required=False, type=pathlib.PurePath)

    ga.add_argument('--ensg2ensp-tsv-path', default='ensg2ensp.tsv',
                    required=False, type=pathlib.PurePath)
    ga.add_argument('--ensg2ensp-oct2014-tsv-path', default='ensg2ensp_oct2014.tsv',
                    required=False, type=pathlib.PurePath)
    ga.add_argument('--ensg2ensp-grch37-tsv-path', default='ensg2ensp_grch37.tsv',
                    required=False, type=pathlib.PurePath)

    ga.add_argument('--umls-mrconso-rrf-path', default='MRCONSO.RRF',
                    required=False, type=pathlib.PurePath)
    ga.add_argument('--string-protein-links-path',
                    default='9606.protein.links.full.v11.5.txt.gz',
                    required=False, type=pathlib.PurePath)

    ga.add_argument('--gwasatlas-path', default='gwasATLAS_v20191115.txt.gz',
                    required=False, type=pathlib.PurePath)
    ga.add_argument('--gwasatlas-magma-path', default='gwasATLAS_v20191115_magma_P.txt.gz',
                    required=False, type=pathlib.PurePath)

    if args.subcommand == 'train':
        create_train_dataset(args)

    if args.subcommand == 'gwasatlas':
        create_gwasatlas_dataset(args)


if __name__ == '__main__':
    main()
