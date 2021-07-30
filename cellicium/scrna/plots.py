import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scvelo.plotting.utils as scvpu
import numpy as np
import pandas as pd

def create_color_gradient(adata, color_gradients):
    color = None
    scatter_kwargs = {'color_gradients': color_gradients, 'size': None, 'palette': 'plasma'}
    vals, names, color, scatter_kwargs = scvpu.gets_vals_from_color_gradients(
        adata, color, **scatter_kwargs
    )
    # print(vals)
    # print(names)
    # print(color)
    cols = zip(adata.obs[color].cat.categories, adata.uns[f"{color}_colors"])
    c_colors = {cat: col for (cat, col) in cols}
    sorted_idx = np.argsort(vals, 1)[:, ::-1][:, :2]
    for id0 in range(len(names)):
        for id1 in range(id0 + 1, len(names)):
            cmap = scvpu.rgb_custom_colormap(
                [c_colors[names[id0]], "white", c_colors[names[id1]]],
                alpha=[1, 0, 1],
            )
#            mkwargs.update({"color_map": cmap})
            c_vals = np.array(vals[:, id1] - vals[:, id0]).flatten()
            c_bool = np.array([id0 in c and id1 in c for c in sorted_idx])
            return c_vals[c_bool]
            # print(c_vals)
            # print(c_bool)
            # if np.sum(c_bool) > 1:
            #     _adata = adata[c_bool] if np.sum(~c_bool) > 0 else adata
            #     mkwargs["color"] = c_vals[c_bool]
            #     ax = scatter(
            #         _adata, ax=ax, **mkwargs, **scatter_kwargs, **kwargs
            #     )


#Show in 3D
def plot_scatter_3d(adata, basis = 'pca', color = None, color_gradients = None, show = True, fig = None):
    #colors = matplotlib.rcParams["axes.prop_cycle"]
#     expr_umap = pd.DataFrame(expr_data.obsm['X_umap_3d'], columns = ['x', 'y', 'z'])
#     expr_umap['group'] = expr_data.obs['clusters'].values

    points = pd.DataFrame(adata.obsm['X_pca'][:, :3], columns = ['x', 'y', 'z'])

    if color is not None:
        color = adata.obs[color]
    elif color_gradients is not None:
        color = create_color_gradient(adata, color_gradients)
    else:
        color = 'blue'

    if fig is None:
        fig = go.Figure(layout = dict(width = 1200, height = 800))

#    for g, ind in groups.items():
#         series_data = expr_umap[expr_umap.group == g]
    fig.add_trace(
        go.Scatter3d(
            x = points.x, y = points.y, z = points.z,
            mode = 'markers', marker = dict(size = 4, color = color)))
    if show:
        fig.show()
    else:
        return fig

def plot_arrows_3d(adata, arrows, color_gradients = None, show = True, fig = None, directions = None):
    if fig is None:
        fig = go.Figure(layout = dict(width = 1200, height = 800))

#    plot_scatter_3d(adata, color_gradients = color_gradients, show = False, fig = fig)

    points = pd.DataFrame(adata.obsm['X_pca'][:, :3], columns = ['x', 'y', 'z'])
    points.u = adata.obsm[arrows][:, 0]
    points.v = adata.obsm[arrows][:, 1]
    points.w = adata.obsm[arrows][:, 2]
    color = create_color_gradient(adata, color_gradients) if color_gradients is not None else 'blue'
    fig.add_trace(
        go.Cone(
            x = points.x, y = points.y, z = points.z,
            u = points.u, v = points.v, w = points.w,
            sizemode = "absolute", sizeref = 1,
            customdata = adata.obs.loc[:, ['phase']],
            hovertemplate='phase: %{customdata[0]} <br>',
        )
    )

    if directions is not None:
        for i, direction in enumerate(directions):
            axis_data = pd.DataFrame({'x': [0, direction[0]], 'y': [0, direction[1]], 'z': [0, direction[2]]})
            fig.add_trace(
                go.Scatter3d(
                    x = axis_data.x, y = axis_data.y, z = axis_data.z,
                    marker = dict(color = i)
                )
            )

    # if start_dir is not None:
    #     start_dir_data = pd.DataFrame({'x': [0, start_dir[0]], 'y': [0, start_dir[1]], 'z': [0, start_dir[2]]})
    #     fig.add_trace(
    #         go.Scatter3d(
    #             x = start_dir_data.x, y = start_dir_data.y, z = start_dir_data.z,
    #             marker = dict(color = 'blue')
    #         )
    #     )

    if show:
        fig.show()
    else:
        return fig


def compare_series(series, prefix = 'Component', components = None, n_comp = None):
    if n_comp is None:
        n_comp = series[0]['y'].shape[1]
    if components is None:
        components = [f'{prefix}{i_comp + 1}' for i_comp in range(n_comp)]
    if n_comp > 5:
        n_row = int(np.ceil(n_comp / 5))
        fig, axes = plt.subplots(n_row, 5, figsize = (15, 5 * n_row))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, n_comp, figsize = (15, 5))
    for i_comp in range(n_comp):
        for i_ser in range(len(series)):
            x = series[i_ser]['x']
            y = series[i_ser]['y'][:, i_comp]
            lab = series[i_ser].get('label', f'Series {i_ser}')
            axes[i_comp].scatter(x, y, label = lab, s = 2)
        axes[i_comp].legend()
        axes[i_comp].set_title(components[i_comp])
