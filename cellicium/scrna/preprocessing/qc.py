import numpy as np
import scanpy as sc
import scipy.sparse as scsp

def qc_metrics(adata):
    # Quality control - calculate QC covariates
    adata.obs['n_counts'] = adata.X.sum(axis = 1)
    #adata.obs['log_counts'] = np.log(adata.obs['n_counts'])
    adata.obs['n_genes'] = (adata.X > 0).sum(axis = 1)
    adata.obs['n_counts_unspliced'] = adata.layers['unspliced'].sum(axis = 1)
    adata.obs['n_genes_unspliced'] = (adata.layers['unspliced'] > 0).sum(axis = 1)
    mt_gene_mask = adata.var_names.str.startswith('mt-') | adata.var_names.str.startswith('MT-')
    adata.var['is_mitochondrial'] = mt_gene_mask
    num_mt_genes = np.sum(mt_gene_mask)
    print(f"Num mitochondrial genes {num_mt_genes}")
    print(f"Mean counts per cell {np.mean(adata.obs['n_counts'])}")
    print(f"SD counts per cell {np.std(adata.obs['n_counts'])}")
    print(f"Mean genes per cell {np.mean(adata.obs['n_genes'])}")
    #[gene.startswith('mt-') for gene in adata.var_names]
    # if (np.any(mt_gene_mask)):
    mt_counts = np.asarray(np.sum(adata.X[:, mt_gene_mask], axis = 1)).flatten()
    adata.obs['mt_frac'] = mt_counts / adata.obs['n_counts']

def filter_cells(adata, min_counts = 1500, max_counts = 40000, max_mt_frac = 0.2, min_genes = 700):
    '''Filter cells according to identified QC thresholds:'''

    print('Total number of cells: {:d}'.format(adata.n_obs))

    sc.pp.filter_cells(adata, min_counts = min_counts)
    print('Number of cells after min count filter: {:d}'.format(adata.n_obs))

    sc.pp.filter_cells(adata, max_counts = max_counts)
    print('Number of cells after max count filter: {:d}'.format(adata.n_obs))

    adata = adata[adata.obs['mt_frac'] < max_mt_frac]
    print('Number of cells after MT filter: {:d}'.format(adata.n_obs))

    sc.pp.filter_cells(adata, min_genes = min_genes)
    print('Number of cells after gene filter: {:d}'.format(adata.n_obs))

def filter_genes(adata, min_counts = 20, min_cells = 5):
    sc.pp.filter_genes(adata, min_counts = min_counts)
    sc.pp.filter_genes(adata, min_cells = min_cells)
