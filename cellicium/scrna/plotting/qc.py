import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scanpy as sc
from .utils import figure_grid

def qc_plots(adata):
    sc.pl.scatter(adata, 'n_counts', 'n_genes', color='mt_frac')
    fg = figure_grid(n_row= 2, n_col= 2)
    sb.distplot(adata.obs['n_counts'], bins = 100, kde=False, ax = next(fg))
    sb.distplot(adata.obs['n_genes'], bins = 100, kde=False, ax = next(fg))
    sb.distplot(adata.obs['n_counts_unspliced'], bins = 100, kde=False, ax = next(fg))
    sb.distplot(adata.obs['n_genes_unspliced'], bins = 100, kde=False, ax = next(fg))
