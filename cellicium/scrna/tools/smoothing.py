import numpy as np
import scipy.signal as sig
import scanpy as sc

def bin_smooth(t, X, n_bins = 50):
    dt = (t[-1] - t[0])  / n_bins
    X_smooth = np.zeros((n_bins, X.shape[1]))
    for i in range(n_bins):
        t_start = i * dt
        t_end = (i + 1) * dt
        bin_filter = (t >= t_start) & (t < t_end)
        X_smooth[i, :] = np.mean(X[bin_filter, :], axis = 0)
    return (np.linspace(0, t[-1], n_bins), X_smooth)

def low_pass_filter(adata : sc.AnnData, time_col, type = 'median', n = 9, layer = None, layer_added = 'X_low_pass'):
        '''This only works because median is nonlinear filter and the time distance is irrelevant. Maybe should be corrected'''
        filtered_data = np.zeros(adata.shape)
        time_order = np.argsort(adata.obs[time_col])
        data = adata.X if layer in (None, 'X') else adata.layers[layer]
        for i, gene in enumerate(adata.var.index.values):
            gene_expr = data[time_order, i].flatten()
            gene_expr = np.concatenate([gene_expr[-n-1:], gene_expr, gene_expr[:n]])
            gene_expr = sig.medfilt(gene_expr, kernel_size = n)
            gene_expr = gene_expr[n+1:-n]
            filtered_data[time_order, i] = gene_expr
        adata.layers[layer_added] = filtered_data
        sc.logging.info(f'Added layer {layer_added}')
