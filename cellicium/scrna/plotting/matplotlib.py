import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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