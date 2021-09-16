import numpy as np
import pandas as pd
import scanpy as sc
import plotly.graph_objects as go
from IPython.display import display
import collections.abc as cabc

def extract_columns(adata : sc.AnnData, col_list : list, axis, layers):
    result_data = {}
    if (layers in ['X', None]) or (isinstance(layers, str) and layers in adata.layers.keys()):
        layers = [layers] * len(col_list)
    elif isinstance(layers, cabc.Collection):
        if len(layers) != len(col_list):
            raise ValueError(f'Number of elements of `layers` must be 1 or {len(col_list)}')
    else:
        raise ValueError('Incorrect `layers` parameter')

    for col_name, layer in zip(col_list, layers):
        if (col_name in adata.var.columns) or (col_name in adata.obs.index):
            if axis is None:
                axis = 'var'
            elif axis != 'var':
                raise ValueError(f'Inconsistent data direction for field {col_name}')

            result_data[col_name] = adata.var_vector(col_name, layer = layer)

        elif (col_name in adata.obs.columns) or (col_name in adata.var.index):
            if axis is None:
                axis = 'obs'
            elif axis != 'obs':
                raise ValueError(f'Inconsistent data direction for field {col_name}')

            result_data[col_name] = adata.obs_vector(col_name, layer = layer)

        else:
            raise ValueError(f'Cannot find field {col_name}')

    if axis == 'var':
        index = adata.var.index
    else:
        index = adata.obs.index

    return pd.DataFrame(data = result_data, index = index)

def scatter(adata : sc.AnnData, x : str, y : str, axis : str = None, annotation = None, layers = None,
            fig = None, show = None, **kwargs):
    if annotation is None:
        annotation = []
    elif not isinstance(annotation, list):
        annotation = [annotation]
    all_columns = [x, y] + annotation
    data = extract_columns(adata, all_columns, axis = axis, layers = layers)

    if fig is None:
        fig = go.Figure(layout = dict(width = 1200, height = 800))
    else:
        show = False

    color = kwargs.pop('color', "#0000FF")
    def format(x):
        if isinstance(x, (float, np.floating)):
            return f"{x:.2f}"
        else:
            return x

    annotation_text = ["<br>".join(
                            [f'{col}: {format(data[col][irow])}' for col in all_columns])
                        for irow in range(data.shape[0])]
    hovertemplate = '%{customdata.index}<br>%{text}' #+ '<br>'.join(['%{customdata.' + col + '}' for col in all_columns]),
    customdata = [{'index': data.index.values[i]}  for i in range(data.shape[0])]
    scatter_trace = \
        go.Scattergl(
            x = data[x], y = data[y], customdata = customdata,
            hovertemplate = hovertemplate,
            text = annotation_text,
            mode = 'markers', marker = dict(size = 4, color = color))

    trace_row = kwargs.pop('row', None)
    trace_column = kwargs.pop('col', None)
    if trace_row and trace_column:
        subplot_id = {
            'row': trace_row,
            'col': trace_column
        }
    else:
        subplot_id = {}

    if trace_row and trace_column:
        fig.add_trace(scatter_trace, **subplot_id)
    else:
        fig.add_trace(scatter_trace, **subplot_id)


    title = kwargs.pop('title', '')
    x_label = kwargs.pop('x_label', x)
    y_label = kwargs.pop('y_label', y)

    # fig.update_layout(
    #     title = title,
    #     xaxis_title = x_label,
    #     yaxis_title = y_label
    # )

    fig.update_xaxes(title_text = x_label, **subplot_id)
    fig.update_yaxes(title_text = y_label, **subplot_id)
    #, showgrid = False
    #fig.update_xaxes(title_text = "xaxis 4 title", type="log", row=2, col=2)

    # Log scales
    log_x, log_y = kwargs.pop('log_scale',  (False, False))
    if log_x:
        fig.update_xaxes(type = "log", **subplot_id)
    if log_y:
        fig.update_yaxes(type = "log", **subplot_id)

    # Click handler
    on_click = kwargs.pop('on_click', None)
    if on_click:
        def on_point_click(trace, points, selector):
            on_click(trace, points, selector)
        # print("Installing click handler")
        #scatter_trace.on_click(on_click)
        fig.data[-1].on_click(on_point_click)

    # if show:
    #     fig.show()
    # else:
    #     return fig
    return fig

#Show in 3D
def scatter_3d(adata, basis = 'pca', color = None, color_gradients = None, show = True, fig = None):
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

def arrows_3d(adata, arrows, color_gradients = None, show = True, fig = None, directions = None):
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
