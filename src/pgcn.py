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

import csv
import itertools
import os
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Tuple, Container, Collection

import h5py
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.utils as pyg_utils
import tqdm
from sklearn import metrics
from torch_geometric import seed_everything

import dataset
from best_model import best_prediction
from model import apk, bedroc_score

# TODO: get these files

# This is from UMLS Metathesaurus version 2019AA
# https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html#2019AA
umls_mrconso_rrf_path = 'MRCONSO.RRF'

# DisGeNet version 7 https://www.disgenet.org/downloads
disgenet_db_path = 'disgenet_2020.db'

# Obtainable from https://github.com/liyu95/Disease_gene_prioritization_GCN
pgcn_gene_phenes_path = 'genes_phenes.mat'
pgcn_prediction_path = 'prediction.npy'

# also edit this if needed:
best_model_timestamp = "2022-10-29T12:30:30+00:00"


def get_hgnc2ncbi(hgnc2ncbi_tsv_path='hgnc2ncbi.tsv'):
    query_xml = \
        '<?xml version="1.0" encoding="UTF-8"?>' \
        '<!DOCTYPE Query>' \
        '<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" datasetConfigVersion="0.6">' \
        '<Dataset name="hsapiens_gene_ensembl" interface="default">' \
        '<Attribute name="hgnc_id"/>' \
        '<Attribute name="entrezgene_id"/>' \
        '</Dataset>' \
        '</Query>'

    if not os.path.exists(hgnc2ncbi_tsv_path):
        url = f'https://www.ensembl.org/biomart/martservice?query=' + urllib.parse.quote(query_xml)
        urllib.request.urlretrieve(url, hgnc2ncbi_tsv_path)

    hgnc2ncbi_entries: set[Tuple[int, int]] = set()
    with open(hgnc2ncbi_tsv_path, 'rt') as tsv:
        for row in csv.DictReader(tsv, delimiter='\t'):
            if row['NCBI gene (formerly Entrezgene) ID'] and row['HGNC ID']:
                hgnc_id = int(row['HGNC ID'][5:])
                ncbi_id = int(row['NCBI gene (formerly Entrezgene) ID'])
                hgnc2ncbi_entries.add((hgnc_id, ncbi_id))
    hgnc2ncbi = defaultdict(set)
    for hgnc, ncbi in hgnc2ncbi_entries:
        hgnc2ncbi[hgnc].add(ncbi)
    return hgnc2ncbi


def get_cui2omim(mrconso_rrf_path='MRCONSO.RRF'):
    cui2omim = defaultdict(set)
    with open(mrconso_rrf_path, 'rt') as mrconso:
        reader = csv.reader(mrconso, delimiter="|")
        for row in tqdm.tqdm(reader, total=14608809):
            # docs: https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr
            cui, lat, ts, lui, stt, sui, ispref, aui, saui, scui, sdui, sab, tty, code, str_, srl, suppress, cvf, _ = row
            if sab == 'OMIM':
                if not code.startswith('MTHU'):
                    if '.' in code:
                        code = code.split('.')[0]
                    cui2omim[cui].add(int(code))
    return cui2omim


def get_ncbi2ensp(ensp2ncbi_tsv_path='ensp2ncbi.tsv', ensp2ncbi_oct2014_tsv_path='ensp2ncbi_oct2014.tsv'):
    query_xml = \
        '<?xml version="1.0" encoding="UTF-8"?>' \
        '<!DOCTYPE Query>' \
        '<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" datasetConfigVersion="0.6">' \
        '<Dataset name="hsapiens_gene_ensembl" interface="default">' \
        '<Attribute name="ensembl_peptide_id"/>' \
        '<Attribute name="entrezgene_id"/>' \
        '</Dataset>' \
        '</Query>'

    if not os.path.exists(ensp2ncbi_tsv_path):
        url = f'https://www.ensembl.org/biomart/martservice?query=' + urllib.parse.quote(query_xml)
        urllib.request.urlretrieve(url, ensp2ncbi_tsv_path)

    ensp2ncbi_entries: set[Tuple[int, int]] = set()
    with open(ensp2ncbi_tsv_path, 'rt') as tsv:
        for row in csv.DictReader(tsv, delimiter='\t'):
            if row['NCBI gene (formerly Entrezgene) ID'] and row['Protein stable ID'].startswith('ENSP'):
                ensembl_peptide_id = int(row['Protein stable ID'][4:])
                ncbi_id = int(row['NCBI gene (formerly Entrezgene) ID'])
                ensp2ncbi_entries.add((ensembl_peptide_id, ncbi_id))

    query_xml = \
        '<?xml version="1.0" encoding="UTF-8"?>' \
        '<!DOCTYPE Query>' \
        '<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" datasetConfigVersion="0.6">' \
        '<Dataset name="hsapiens_gene_ensembl" interface="default">' \
        '<Attribute name="ensembl_peptide_id"/>' \
        '<Attribute name="entrezgene"/>' \
        '</Dataset>' \
        '</Query>'

    if not os.path.exists(ensp2ncbi_oct2014_tsv_path):
        url = f'https://oct2014.archive.ensembl.org/biomart/martservice?query=' \
              + urllib.parse.quote(query_xml)
        urllib.request.urlretrieve(url, ensp2ncbi_oct2014_tsv_path)

    ensp2ncbi_oct2014_entries: set[Tuple[int, int]] = set()
    with open(ensp2ncbi_oct2014_tsv_path, "rt") as tsv:
        for row in csv.DictReader(tsv, delimiter='\t'):
            if row['EntrezGene ID'] and row['Ensembl Protein ID'].startswith('ENSP'):
                ensembl_peptide_id = int(row['Ensembl Protein ID'][4:])
                ncbi_id = int(row['EntrezGene ID'])
                ensp2ncbi_oct2014_entries.add((ensembl_peptide_id, ncbi_id))

    ncbi2ensp = defaultdict(set)
    for ensp, ncbi in ensp2ncbi_entries:
        ncbi2ensp[ncbi].add(f"9606.ENSP{ensp:011}")
    for ensp, ncbi in ensp2ncbi_oct2014_entries:
        ncbi2ensp[ncbi].add(f"9606.ENSP{ensp:011}")

    return ncbi2ensp


def load_pgcn_metadata(gene_phenes_path='genes_phenes.mat'):
    with h5py.File(gene_phenes_path, 'r') as f:
        num_genes = int(f['numGenes'][0].item())
        species = [''.join(chr(int(x)) for x in y) for y in np.array(f['GP_SPECIES']).T]
        hs_index = species.index('Hs')  # homo sapiens

        phene_ids = np.array(f[f['pheneIds'][hs_index][0]], dtype=int).flatten()
        gene_ids = np.array(f['geneIds'], dtype=int).flatten()

        gene_phene = f[f['GenePhene'][hs_index][0]]
        gene_phene_adj = sp.csc_matrix((np.array(gene_phene['data']),
                                        np.array(gene_phene['ir']), np.array(gene_phene['jc'])),
                                       shape=(num_genes, len(phene_ids)))

    gene_phene_adj = gene_phene_adj.tocoo()
    return gene_ids, phene_ids, gene_phene_adj


def get_disgenet_ncbi2omim(disgenet_hgnc2cui, hgnc2ncbi, cui2omim):
    disgenet_ncbi2omim = nx.DiGraph()
    for g, d in disgenet_hgnc2cui.edges:
        if g in hgnc2ncbi and d in cui2omim:
            score = disgenet_hgnc2cui.edges[g, d]["score"]
            for ncbi in hgnc2ncbi[g]:
                for omim in cui2omim[d]:
                    if not disgenet_ncbi2omim.has_edge(ncbi, omim):
                        disgenet_ncbi2omim.add_edge(ncbi, omim)
                        disgenet_ncbi2omim.edges[ncbi, omim]["score"] = score
                    else:
                        prev_score = disgenet_ncbi2omim.edges[ncbi, omim]["score"]
                        disgenet_ncbi2omim.edges[ncbi, omim]["score"] = max(prev_score, score)
    return disgenet_ncbi2omim


def main():
    seed_everything(0)

    hgnc2ncbi = get_hgnc2ncbi()
    cui2omim = get_cui2omim(umls_mrconso_rrf_path)
    ncbi2ensp = get_ncbi2ensp()
    omim2cui = defaultdict(set)
    for cui, omims in cui2omim.items():
        for omim in omims:
            omim2cui[omim].add(cui)

    disgenet_hgnc2cui = dataset.create_disgenet_graph(disgenet_db_path)
    disgenet_ncbi2omim = get_disgenet_ncbi2omim(disgenet_hgnc2cui, hgnc2ncbi, cui2omim)

    gene_ids, phene_ids, gene_phene_adj = load_pgcn_metadata(pgcn_gene_phenes_path)
    ncbi2pgcn_gene_index = {ncbi: i for i, ncbi in enumerate(gene_ids)}
    omim2pgcn_pheno_index = {omim: i for i, omim in enumerate(phene_ids)}

    training_data = [(gene_ids[i], phene_ids[j]) for i, j in zip(gene_phene_adj.row, gene_phene_adj.col)]
    disgenet_ncbi2omim.remove_edges_from(training_data)

    best_pred, train_set = best_prediction(best_model_timestamp)

    ensp2gene_index = {ensp: i for i, ensp in enumerate(best_pred.gene_ids)}
    cui2pheno_index = {cui: i for i, cui in enumerate(best_pred.phenotype_ids)}

    ncbi2pred_gene_index = defaultdict(set)
    for ncbi in gene_ids:
        if ncbi in ncbi2ensp:
            for ensp in ncbi2ensp[ncbi]:
                if ensp in ensp2gene_index:
                    ncbi2pred_gene_index[ncbi].add(ensp2gene_index[ensp])

    omim2pred_pheno_index = defaultdict(set)
    for omim in phene_ids:
        if omim in omim2cui:
            for cui in omim2cui[omim]:
                if cui in cui2pheno_index:
                    omim2pred_pheno_index[omim].add(cui2pheno_index[cui])

    all_edges_pgcn = list()
    high_quality_edges_pgcn = list()
    medium_quality_edges_pgcn = list()
    low_quality_edges_pgcn = list()

    all_edges_pred = list()
    high_quality_edges_pred = list()
    medium_quality_edges_pred = list()
    low_quality_edges_pred = list()

    for ncbi, omim in disgenet_ncbi2omim.edges:
        score = disgenet_ncbi2omim.edges[ncbi, omim]["score"]
        if ncbi not in ncbi2pgcn_gene_index or omim not in omim2pgcn_pheno_index:
            continue
        edge_pgcn = (ncbi2pgcn_gene_index[ncbi], omim2pgcn_pheno_index[omim])
        all_edges_pgcn.append(edge_pgcn)
        if score >= 0.1:
            high_quality_edges_pgcn.append(edge_pgcn)
        if 0.01 < score < 0.1:
            medium_quality_edges_pgcn.append(edge_pgcn)
        if score == 0.01:
            low_quality_edges_pgcn.append(edge_pgcn)

        for gene in ncbi2pred_gene_index[ncbi]:
            for pheno in omim2pred_pheno_index[omim]:
                edge_pred = (gene, pheno)
                all_edges_pred.append(edge_pred)
                if score >= 0.1:
                    high_quality_edges_pred.append(edge_pred)
                if 0.01 < score < 0.1:
                    medium_quality_edges_pred.append(edge_pred)
                if score == 0.01:
                    low_quality_edges_pred.append(edge_pred)

    all_edges_pgcn = np.array(all_edges_pgcn).T
    high_quality_edges_pgcn = np.array(high_quality_edges_pgcn).T
    medium_quality_edges_pgcn = np.array(medium_quality_edges_pgcn).T
    low_quality_edges_pgcn = np.array(low_quality_edges_pgcn).T

    all_edges_pred = np.array(all_edges_pred).T
    high_quality_edges_pred = np.array(high_quality_edges_pred).T
    medium_quality_edges_pred = np.array(medium_quality_edges_pred).T
    low_quality_edges_pred = np.array(low_quality_edges_pred).T

    pgcn_prediction = np.load(pgcn_prediction_path).T

    original_training_edges_pgcn = np.vstack((gene_phene_adj.row, gene_phene_adj.col))

    original_training_edges_pred = list()
    for ncbi, omim in training_data:
        for gene in ncbi2pred_gene_index[ncbi]:
            for pheno in omim2pred_pheno_index[omim]:
                original_training_edges_pred.append((gene, pheno))
    original_training_edges_pred = np.array(original_training_edges_pred).T

    pgcn_edge_sets = (original_training_edges_pgcn, high_quality_edges_pgcn,
                      medium_quality_edges_pgcn, low_quality_edges_pgcn)
    pred_edge_sets = (original_training_edges_pred, high_quality_edges_pred,
                      medium_quality_edges_pred, low_quality_edges_pred)
    descriptions = ("PGCN training set", "DisGeNet high quality", "DisGeNet medium quality", "DisGeNet low quality")

    for pos_edges_pgcn, pos_edges_pred, desc in zip(pgcn_edge_sets, pred_edge_sets, descriptions):

        params = zip(("PGCN", "Pred"), (all_edges_pgcn, all_edges_pred),
                     ((len(gene_ids), len(phene_ids)), (len(best_pred.gene_ids), len(best_pred.phenotype_ids))),
                     (pos_edges_pgcn, pos_edges_pred),
                     (pgcn_prediction, best_pred.scores))
        for name, all_edges, num_nodes, pos_edges, scores in params:
            neg_edges = pyg_utils.negative_sampling(
                torch.from_numpy(all_edges), num_nodes, pos_edges_pgcn.shape[1]
            ).numpy()
            y_pos = scores[pos_edges[0], pos_edges[1]]
            y_neg = scores[neg_edges[0], neg_edges[1]]
            y_pred = np.hstack((y_pos, y_neg))
            y_true = np.hstack((np.ones(pos_edges.shape[1]), np.zeros(neg_edges.shape[1])))

            roc_sc = metrics.roc_auc_score(y_true, y_pred)
            aupr_sc = metrics.average_precision_score(y_true, y_pred)
            bedroc_sc = bedroc_score(y_true, y_pred)

            actual = list(np.where(y_true == 1)[0])
            predicted = [i for i, score in sorted(enumerate(y_pred), key=lambda x: x[1], reverse=True)]
            apk_sc = apk(actual, predicted, k=200)

            print(f"{name} tested on {pos_edges.shape[1]} {desc} edges.")
            print(f"AUROC:{roc_sc:.3f} AUPRC:{aupr_sc:.3f} AP@200:{apk_sc:.3f} BEDROC:{bedroc_sc:.3f}")
            print()


if __name__ == '__main__':
    main()
