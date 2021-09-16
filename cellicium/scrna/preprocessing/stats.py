import numpy as np
from scipy.sparse import issparse
import scanpy as sc

def zero_statistics(adata : sc.AnnData):
    cdata = adata.X
    if issparse(cdata):
        cdata = np.asarray(cdata.todense())
    num_zeros = np.sum(cdata == 0)
    frac_zeros = num_zeros / np.prod(np.shape(cdata))
    print(f"Global fraction of zeros {frac_zeros:.3f}")
    # Cells statistics
    adata.obs['frac_nz'] = np.sum(cdata > 0, axis = 1) / adata.shape[1]
    adata.obs['frac_zeros'] = 1 - adata.obs['frac_nz']
    # Gene statistics
    adata.var['frac_nz'] = np.sum(cdata > 0, axis = 0) / adata.shape[0]
    adata.var['frac_zeros'] = 1 - adata.var['frac_nz']

def gene_statistics(adata : sc.AnnData, layer = None, prefix = None, peak_vals = None):
    if layer in (None, 'X'):
        data = adata.X
    else:
        data = adata.layers[layer]
        if prefix is None:
            prefix = layer.lower()

    adata.var[f'{prefix}_mean'] = np.mean(data, axis = 0)
    adata.var[f'{prefix}_sd'] = np.std(data, axis = 0)
    if peak_vals is not None:
        adata.var[f'{prefix}_z'] = (peak_vals - adata.var[f'{prefix}_mean']) / adata.var[f'{prefix}_sd']

def estimate_gene_peaks(adata : sc.AnnData, layer = None, peak_est_n = 1):
    num_genes = adata.shape[1]
    gene_peak_est = np.zeros(num_genes)
    if layer in (None, 'X'):
        data = adata.X
    else:
        data = adata.layers[layer]

    for i in range(num_genes):
        expr = data[:, i]
        expr_top = np.sort(expr)[-1:-1-peak_est_n:-1]
        gene_peak_est[i] = np.mean(expr_top)
    return gene_peak_est

# def gene_statistics(adata : sc.AnnData, peak_vals = None):
#     adata.var['count_mean'] = np.mean(cdata, axis = 0)
#     # adata.var['count_mean_nz'] = np.sum(cdata, axis = 0) / np.sum(cdata > 0, axis = 0)
#     adata.var['count_sd'] = np.std(cdata, axis = 0)
#
#     if ('X_log_norm' in adata.layers.keys()):
#         X_log_norm = adata.layers['X_log_norm']
#         prefix = 'x_log_norm'
