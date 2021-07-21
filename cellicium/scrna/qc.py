import matplotlib.pyplot as plt
import seaborn as sb
import scanpy as sc


def compute_qc_metrics(adata):
    # Quality control - calculate QC covariates
    adata.obs['n_counts'] = adata.X.sum(1)
    #adata.obs['log_counts'] = np.log(adata.obs['n_counts'])
    adata.obs['n_genes'] = (adata.X > 0).sum(1)
    adata.obs['n_counts_unspliced'] = adata.layers['unspliced'].sum(1)
    adata.obs['n_genes_unspliced'] = (adata.layers['unspliced'] > 0).sum(1)
    mt_gene_mask = adata.var_names.str.startswith('mt-') | adata.var_names.str.startswith('MT-')
    adata.var['is_mitochondrial'] = mt_gene_mask
    #[gene.startswith('mt-') for gene in adata.var_names]
    adata.obs['mt_frac'] = adata.X[:, mt_gene_mask].sum(1)/adata.obs['n_counts']


def qc_plots(adata):
    compute_qc_metrics(adata)
    #sc.pl.scatter(adata, 'n_counts', 'n_genes', color='mt_frac', legend_loc = 'on data')
    #fig, axes = plt.subplots(1, 3, figsize = (15, 5))
    sc.pl.scatter(adata, 'n_counts', 'n_genes', color='mt_frac')
    #p2 = sc.pl.scatter(adata[adata.obs['n_counts']<10000], 'n_counts', 'n_genes', color='mt_frac', ax = axes[1], show = False)
    # Histograms
    # fig, axes = plt.subplots(1, 2, figsize = (15, 5))
    # sb.distplot(adata.obs['n_counts'], bins = 100, kde=False, ax = axes[0])
    # sb.distplot(adata.obs['n_genes'], bins = 100, kde=False, ax = axes[1])
    # With unspliced
    fig, axes = plt.subplots(2, 2, figsize = (15, 10))
    sb.distplot(adata.obs['n_counts'], bins = 100, kde=False, ax = axes[0, 0])
    sb.distplot(adata.obs['n_genes'], bins = 100, kde=False, ax = axes[0, 1])
    sb.distplot(adata.obs['n_counts_unspliced'], bins = 100, kde=False, ax = axes[1, 0])
    sb.distplot(adata.obs['n_genes_unspliced'], bins = 100, kde=False, ax = axes[1, 1])
