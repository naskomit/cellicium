import numpy as np
import matplotlib.pyplot as plt
#import scvelo.plotting.utils as scvpu

# Creates a rectangular grid of figures/axes
def figure_grid(n_col = 5, n_row = None, n_total = None, figsize = None, width = None, row_height = None):
    if n_row is None:
        n_row = int(np.ceil(n_total / n_col))

    if figsize is None:
        if width is None:
            width = 20
        if row_height is None:
            row_height = 10
        figsize = (width, row_height * n_row)
    figure, axes = plt.subplots(n_row, n_col, figsize = figsize, squeeze = False, gridspec_kw = {'hspace': 0.3, 'wspace': 0.4})
    index = 0
    for row in range(n_row):
        for col in range(n_col):
            ax = axes[row, col]
            yield ax

# def create_color_gradient(adata, color_gradients):
#     color = None
#     scatter_kwargs = {'color_gradients': color_gradients, 'size': None, 'palette': 'plasma'}
#     vals, names, color, scatter_kwargs = scvpu.gets_vals_from_color_gradients(
#         adata, color, **scatter_kwargs
#     )
#     # print(vals)
#     # print(names)
#     # print(color)
#     cols = zip(adata.obs[color].cat.categories, adata.uns[f"{color}_colors"])
#     c_colors = {cat: col for (cat, col) in cols}
#     sorted_idx = np.argsort(vals, 1)[:, ::-1][:, :2]
#     for id0 in range(len(names)):
#         for id1 in range(id0 + 1, len(names)):
#             cmap = scvpu.rgb_custom_colormap(
#                 [c_colors[names[id0]], "white", c_colors[names[id1]]],
#                 alpha=[1, 0, 1],
#             )
# #            mkwargs.update({"color_map": cmap})
#             c_vals = np.array(vals[:, id1] - vals[:, id0]).flatten()
#             c_bool = np.array([id0 in c and id1 in c for c in sorted_idx])
#             return c_vals[c_bool]
#             # print(c_vals)
#             # print(c_bool)
#             # if np.sum(c_bool) > 1:
#             #     _adata = adata[c_bool] if np.sum(~c_bool) > 0 else adata
#             #     mkwargs["color"] = c_vals[c_bool]
#             #     ax = scatter(
#             #         _adata, ax=ax, **mkwargs, **scatter_kwargs, **kwargs
#             #     )
#
# def compare_series(series, prefix = 'Component', components = None, n_comp = None):
#     if n_comp is None:
#         n_comp = series[0]['y'].shape[1]
#     if components is None:
#         components = [f'{prefix}{i_comp + 1}' for i_comp in range(n_comp)]
#     if n_comp > 5:
#         n_row = int(np.ceil(n_comp / 5))
#         fig, axes = plt.subplots(n_row, 5, figsize = (15, 5 * n_row))
#         axes = axes.flatten()
#     else:
#         fig, axes = plt.subplots(1, n_comp, figsize = (15, 5))
#     for i_comp in range(n_comp):
#         for i_ser in range(len(series)):
#             x = series[i_ser]['x']
#             y = series[i_ser]['y'][:, i_comp]
#             lab = series[i_ser].get('label', f'Series {i_ser}')
#             axes[i_comp].scatter(x, y, label = lab, s = 2)
#         axes[i_comp].legend()
#         axes[i_comp].set_title(components[i_comp])
