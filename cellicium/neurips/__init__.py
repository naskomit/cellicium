# import numpy as np
# import scipy.sparse as sparse
# import matplotlib.pyplot as plt
# import scanpy as sc
# import seaborn as sb
# import cellicium.scrna as crna
# from cellicium.logging import logger as log
#
# import pandas as pd

from . import preprocessing as pp
from . import tools as tl
from . import deepnmf as deepnmf
from . import vae

from .siamese import SiamesseModel, SiameseModelManager
from .geneclr import MultimodalManager
from . import gmm

# def matrix_min_max(X):
#     i_min = np.argmin(X)
#     i_max = np.argmax(X)
#     row_min, col_min = divmod(i_min, X.shape[1])
#     row_max, col_max = divmod(i_max, X.shape[1])
#     #print(f'{row_min, col_min}: {X[row_min, col_min]}')
#     #print(f'{row_max, col_max}: {X[row_max, col_max]}')
#     return ((row_min, col_min), (row_max, col_max))
#
# def compare_batch_expressions(adata, exclude_vars = [], layer = None):
#     var_selector = ~adata.var.index.isin(exclude_vars)
#     batches = adata.obs['batch'].unique()
#     if layer is None:
#         counts = adata.X
#     else:
#         counts = adata.layers[layer]
#     mean_levels = []
#     excluded_counts = []
#     for batch in batches:
#         batch_selector = adata.obs['batch'] == batch
#         batch_counts = np.asarray(counts[batch_selector, :].todense())
#         mean_in_batch = np.asarray(np.mean(batch_counts, axis = 0))
#         mean_library_size = np.array(np.mean(np.sum(batch_counts, axis = 1)))
#         mean_in_batch = np.log10(mean_in_batch / mean_library_size)
#         mean_levels.append(mean_in_batch)
#     mean_levels = pd.DataFrame(mean_levels, columns = adata.var.index[var_selector], index = batches).T
#     mean_levels = mean_levels.sort_index()
#     #excluded_counts = pd.DataFrame(excluded_counts, columns = adata.var.index[~var_selector], index = batches).T
#     fig = plt.figure(figsize = (15, 25))
#     sb.heatmap(mean_levels, ax = fig.gca(), cmap = 'viridis')
#
# def compute_batch_coefficients(self, df, var_columns, batch_column = 'batch'):
#     log.info('Computing batch coefficients...')
#     import statsmodels.formula.api as smf
#
#     df[batch_column] = pd.Categorical(df[batch_column])
#     batches = df[batch_column].values.categories
#     batch_coefficients = np.zeros((len(var_columns), batches.shape[0]))
#
#     for i, var in enumerate(var_columns):
#         mod = smf.glm(formula = f'{var} ~ {batch_column} - 1', data = df)
#         res = mod.fit()
#         batch_coefficients [i, :] = res.params
#
#     batch_coefficients = pd.DataFrame(data = batch_coefficients, columns = batches, index = var_columns)
#     return batch_coefficients
#
#
#
#
# def plot_top_program_usage(adata, layer = None, n_top = 2):
#     fg = crna.pl.figure_grid(ncol = n_top, nrow = 2)
#     if layer is None:
#         usages = adata.X
#     else:
#         usages = adata.obsm[layer]
#         if (isinstance(usages, pd.DataFrame)):
#             usages = usages.values
#     p1 = pd.Categorical(np.argmax(usages, axis = 1) + 1)
#     usages = np.sort(usages, axis = -1)
#     u1 = usages[:, -1]
#     u1_col = 'tmp_u1' if layer is None else f'tmp_{layer}_u1'
#     p1_col = 'tmp_p1' if layer is None else f'tmp_{layer}_p1'
#     adata.obs[u1_col] = u1
#     adata.obs[p1_col] = p1
#
#
#     for i in range(n_top):
#         ui = usages[:, (-2 - i)]
#         ui_ratio = ui / u1
#         ax = next(fg)
#         ax.set_xlabel(f'U{i + 2} / U1')
#         sb.histplot(ui_ratio, stat = 'probability', bins = np.arange(0, 1, 0.1), ax = ax)
#
#     for i in range(n_top):
#         ui = usages[:, (-2 - i)]
#         adata.obs['tmp_ui_ratio'] = ui / u1
#         ax = next(fg)
#         ax.set_xlabel(f'U{i + 2} / U1')
#         sc.pl.umap(adata, color = 'tmp_ui_ratio', ax = ax, show = False)
#
#     fg = crna.pl.figure_grid(ncol = 2, nrow = 1)
#     sc.pl.umap(adata, color = u1_col, ax = next(fg), show = False)
#     sc.pl.umap(adata, color = p1_col, ax = next(fg), legend_loc = 'on data', show = False)
#
#
#
# def check_expression_recovery(adata):
#     programs = adata.uns['programs']
#     usage = adata.obs.loc[:, programs].values
#     activity = adata.obs['activity_total'].values.reshape((-1, 1))
#     usage = usage * activity
#     genes = adata.var.loc[:, programs].values.T
#     X_recovered = usage @ genes
#     X = np.log(1 + np.asarray(adata.X.todense()))
#     error = np.abs(X_recovered - X)
#     print(X.shape, error.shape)
#     print(type(X), type(error))
#
#     #plt.scatter(X.flatten(), error.flatten())
#     #     ((row_min, col_min), (row_max, col_max)) = matrix_min_max(error)
#     #     print(X[row_max, col_max])
#     #     print(X_recovered[row_max, col_max])
#     #print(type(ratio.reshape(-1)))
#     #sb.histplot(np.asarray(ratio).flatten(), log_scale = (10, 10))
#     return error
#
# # def compute_neighbours(adata, )
#
# def gex_adt_map(gex, adt, method):
#     usages_gex = gex.X
#     activity_gex = gex.obs['activity_total'].values.reshape((-1, 1))
#
#     usages_adt = adt.X
#     activity_adt = adt.obs['activity_total'].values.reshape((-1, 1))
#
#     #     usages_gex = usages_gex * activity_gex / activity_adt
#
#     #usages_adt_raw = adt.X.todense()
#
#     # if method == 'cosine':
#     #     usage_cross = np.zeros((usages_gex.shape[1], usages_adt.shape[1]))
#     #     for i_gex in range(usages_gex.shape[1]):
#     #         for i_adt in range(usages_adt.shape[1]):
#     #             usage_tmp = np.dot(usages_gex[:, i_gex], usages_adt[:, i_adt])
#     #             usage_tmp = usage_tmp / np.linalg.norm(usages_gex[:, i_gex]) / np.linalg.norm(usages_adt[:, i_adt])
#     #             usage_cross[i_gex, i_adt] = usage_tmp
#
#     if method == 'lin_reg':
#         usage_cross_gex_adt, _, _, _ = np.linalg.lstsq(usages_gex, usages_adt)
#         usage_cross_adt_gex, _, _, _ = np.linalg.lstsq(usages_adt, usages_gex)
#         #usage_cross[usage_cross < 0.1] = 0
#
#     elif method == 'lin_reg_activity':
#         usage_cross_gex_adt, _, _, _ = np.linalg.lstsq(usages_gex * activity_gex / activity_adt, usages_adt)
#         usage_cross_adt_gex, _, _, _ = np.linalg.lstsq(usages_adt, usages_gex * activity_gex / activity_adt)
#
#     else:
#         raise ValueError(f'Unknown method {method}')
#
#     return usage_cross_gex_adt, usage_cross_adt_gex
#
#
#
# def plot_heatmap(X, sort = None, ax = None, **kwargs):
#     if sort == 'row':
#         row_max_ind = np.argmax(X.values, axis = 1)
#         X = X.iloc[np.argsort(row_max_ind), :]
#     elif sort == 'col':
#         row_max_ind = np.argmax(X.values, axis = 0)
#         X = X.iloc[:, np.argsort(row_max_ind)]
#
#     figsize = kwargs.pop('figsize', (15, 15))
#     if ax is None:
#         fig = plt.figure(figsize = figsize)
#         ax = fig.gca()
#     sb.heatmap(X, ax = ax, **kwargs)
#
#
# def correspondence_matrix(adata, **kwargs):
#     if 'cmap' not in kwargs:
#         kwargs['cmap'] = 'viridis'
#
#     programs = adata.uns['programs']
#     df = pd.DataFrame(adata.X, columns = adata.var.index, index = adata.obs.index)
#     df = df.join(adata.obs[['cell_type']])
#     fg = crna.pl.figure_grid(ncol = 2, nrow = 1, figsize = (30, 15))
#     ax = next(fg)
#     ax.set_title('Correspondence matrix by mean values')
#
#     corr_table = df.groupby('cell_type')[programs].mean()
#     plot_heatmap(corr_table, sort ='col', ax = ax, **kwargs)
#
#     ax = next(fg)
#     ax.set_title('Correspondence matrix by top values')
#     obs = adata.obs.copy()
#     top_programs = np.argmax(df[programs].values, axis = 1)
#     obs[programs] = 0
#     for i in range(obs.shape[0]):
#         obs[programs[top_programs[i]]][i] = 1
#     corr_table = obs.groupby('cell_type')[adata.uns['programs']].mean()
#     plot_heatmap(corr_table, sort ='col', ax = ax, **kwargs)
#
#
# def sample_mod_data(mod1_data : sc.AnnData, mod2_data: sc.AnnData, n_samples : int = 1000, seed = 12345, shuffle = True):
#     num_obs = mod1_data.n_obs
#     rnd_gen = np.random.default_rng(seed)
#     # rnd_indices = rnd_gen.integers(low = 0, high = num_obs - 1, size = n_samples)
#     rnd_indices = rnd_gen.choice(num_obs, size = n_samples, replace = False)
#     if shuffle:
#         shuffle_ind = rnd_gen.choice(n_samples, size = n_samples, replace = False)
#     else:
#         shuffle_ind = np.arange(n_samples)
#     rnd_indices_2 = rnd_indices[shuffle_ind]
#     pairing_matrix = sparse.csc_matrix((np.ones(n_samples), (np.arange(n_samples), shuffle_ind)), shape = (n_samples, n_samples))
#     pairing_matrix = sc.AnnData(X = pairing_matrix)
#     # print(rnd_indices)
#     # print(rnd_indices_2)
#     # print(pairing_matrix.todense())
#     return mod1_data[rnd_indices, :], mod2_data[rnd_indices_2, :], pairing_matrix
