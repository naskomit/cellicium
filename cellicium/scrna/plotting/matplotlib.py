import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    for row in range(n_row):
        for col in range(n_col):
            ax = axes[row, col]
            yield ax


# def figure_grid_1var(adata : sc.AnnData, )


def get_or_create_axis(kwargs):
    ax = kwargs.pop('ax', None)
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    return ax


def text_plot(df : pd.DataFrame, x : str, y : str , **kwargs):
    fontsize = kwargs.pop('fontsize', 8)
    x_vals = df[x]
    y_vals = df[y]

    ax = get_or_create_axis(kwargs)
    ax.set_xlim(-3, df.shape[0] + 3)
    dy = np.max(y_vals) - np.min(y_vals)
    ax.set_ylim(np.min(y_vals) - 0.2 * dy, np.max(y_vals) + 0.2 * dy)

    for i_val, val_name in enumerate(x_vals):
        ax.text(
            i_val,
            y_vals[i_val],
            val_name,
            rotation = 'vertical',
            verticalalignment = 'bottom',
            horizontalalignment = 'center',
            fontsize = fontsize
        )