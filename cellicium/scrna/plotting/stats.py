import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scanpy as sc
import scipy.stats as stats
import tensorflow_probability as tfp
from .utils import figure_grid

def zero_statistics(adata):
    fg = figure_grid(n_row= 2, n_col= 2)

    ax = next(fg)
    sb.histplot(adata.obs['frac_zeros'], bins = 20, ax = ax)
    ax.set_title("Fraction zeros per cell")

    ax = next(fg)
    sb.histplot(adata.var['frac_zeros'], bins = 20, ax = ax)
    ax.set_title("Fraction zeros per gene")

    ax = next(fg)
    ax.scatter(adata.obs['n_counts'], adata.obs['frac_nz'], s = 2)
    ax.set_xlabel('n_counts')
    ax.set_ylabel('frac_nz')
    ax.set_title('Non-zero fraction vs counts vs (per cell)')

    ax = next(fg)
    ax.scatter(adata.var['count_mean'], adata.var['frac_nz'], s = 2)
    ax.set_xlabel('count_mean')
    ax.set_ylabel('frac_nz')
    ax.set_xscale('log')
    ax.set_title('Non-zero fraction vs mean expr  (per gene)')

    # ax = next(fg)
    # ax.scatter(adata.var['count_mean'], adata.var['count_mean_nz'], s = 2)
    # ax.set_xlabel('count_mean')
    # ax.set_ylabel('count_mean_nz')
    # ax.set_xscale('log')
    # ax.set_yscale('log')

# def peak_count_distribution(adata, fg, peak_est_n, stat):
#     num_genes = adata.shape[1]
#     gene_peak_est = np.zeros(num_genes)
#     for i in range(num_genes):
#         expr = adata.X[:, i]
#         expr_top = np.sort(expr)[-1:-1-peak_est_n:-1]
#         gene_peak_est[i] = np.mean(expr_top)
#
#     ax = next(fg)
#     sb.histplot(gene_peak_est + 1, bins = 100, stat = stat, log_scale = (10, 10), ax = ax)
#     ax.set_xlabel('peak_count (est)')
#     ax.set_ylabel(stat)
#     ax.set_title(f'Distribution of estimated peak counts (by top {peak_est_n}) ')
#     ax.grid(True)
#
#     # Estimate power law
#     # alpha, loc, scale = stats.gamma.fit(gene_peak_est, floc = 0, fscale = 100)
#     # print(alpha, loc, scale)
#     x = 10 ** np.linspace(0, 2.5, 100)
#
#     pdf_coeff = 1.0 if stat == 'probability' else gene_peak_est.shape[0]
#     pdf = pdf_coeff * stats.gamma(0.2, loc = 0, scale = 100).pdf(x)
#     #pdf = stats.gamma(alpha, loc = loc, scale = scale).pdf(x)
#     ax.plot(x, pdf, 'r-', lw = 2, label = 'estimated powerlaw')
#     pdf = pdf_coeff * stats.gamma(0.2, loc = 0, scale = 50).pdf(x)
#     #pdf = stats.gamma(alpha, loc = loc, scale = scale).pdf(x)
#     ax.plot(x, pdf, 'b-', lw = 2, label = 'estimated powerlaw')
#
#     ax = next(fg)
#     ax.scatter(gene_peak_est, adata.var['count_mean'], s = 2)
#     # sb.jointplot(x = gene_peak_est, y = adata.var['count_mean'], ax = ax)
#     ax.set_xlabel('peak_count (est)')
#     ax.set_ylabel('mean count')
#     ax.set_title('Mean count across cells vs est. peak count')
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.grid(True)
#
#     ax = next(fg)
#     ax.scatter(adata.var['count_mean'], adata.var['count_sd'], s = 2)
#     ax.set_xlabel('count_mean')
#     ax.set_ylabel('count_sd')
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_title('Mean count vs SD (per gene)')
#     ax.grid(True)
#
#     ax = next(fg)
#     ax.scatter(gene_peak_est, adata.var['count_sd'], s = 2)
#     ax.set_xlabel('peak_count (est)')
#     ax.set_ylabel('count_sd')
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_title('Peak count (est) vs SD (per gene)')
#     ax.grid(True)
#
#     return gene_peak_est

# def gene_statistics(adata : sc.AnnData, peak_est_n = 5,
#                     stat = 'probability', save = []):
#     fg = figure_grid(nrow = 3, ncol = 2)
#
#     ax = next(fg)
#     sb.histplot(1e-3 + adata.var['count_mean'], bins = 100, stat = stat, log_scale = (10, 10), ax = ax)
#     ax.set_xlabel('mean count')
#     ax.set_ylabel(stat)
#     ax.set_title(f'Distribution of mean count')
#     ax.grid(True)
#
#     gene_peak_est = peak_count_distribution(adata, fg = fg, peak_est_n = peak_est_n, stat = stat)
#     if 'peak_est' in save:
#         adata.var['peak_est'] = gene_peak_est
