import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_or_create_axis(kwargs):
    ax = kwargs.pop('ax', None)
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    return ax

def text_plot(df : pd.DataFrame, x : str, y : str, **kwargs):
    fontsize = kwargs.pop('fontsize', 8)

    ax = get_or_create_axis(kwargs)
    x_vals = df[x]
    y_vals = df[y]
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