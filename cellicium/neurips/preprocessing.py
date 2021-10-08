import numpy as np
import pandas as pd
import scanpy as sc
import os
from IPython.display import display
from cellicium.logging import logger as log
import cellicium.scrna as crna
import seaborn as sb

base_paths = {
 'benchmark': '/home/jovyan/neurips/datasets/cite/'
}

def load_data(name, batches = None):
    if name == 'benchmark':
        base_path = base_paths[name]
        gex_path = os.path.join(base_path, 'cite_gex_processed_training.h5ad')
        adt_path = os.path.join(base_path, 'cite_adt_processed_training.h5ad')
        # log.info(os.path.abspath(gex_path))
        gex_data = sc.read_h5ad(gex_path)
        adt_data = sc.read_h5ad(adt_path)
        # No total_counts is present in gex_data
        gex_data.obs['total_counts'] = np.sum(gex_data.layers['counts'], axis = 1)
        gex_data.obs['site'] = gex_data.obs['batch'].str.slice(0, 2)
        gex_data.obs['donor'] = gex_data.obs['batch'].str.slice(2, 4)
        adt_data.obs['site'] = adt_data.obs['batch'].str.slice(0, 2)
        adt_data.obs['donor'] = adt_data.obs['batch'].str.slice(2, 4)

        gex_data.uns['modality'] = 'GEX'
        adt_data.uns['modality'] = 'ADT'
        return {'GEX': gex_data, 'ADT': adt_data}
    else:
        raise ValueError(f'Unknown dataset {name}')

def datasets_info(adata_dict):
    rows = []
    index = []
    for name, adata in adata_dict.items():
        print(f"============= {name} ===============")
        info = {
            'Number of cells': adata.obs.shape[0],
            'Number of batches': adata.obs['batch'].unique().shape[0],
            'Number of clusters': adata.obs['cell_type'].unique().shape[0]
        }
        rows.append(info)
        index.append(name)
        batch_num_cells = adata.obs['batch'].value_counts().rename('num_cells')
        batch_med_counts = adata.obs.groupby('batch').median()['total_counts'].rename('median_count')
        batch_info = pd.merge(left = batch_num_cells, right = batch_med_counts, left_index=True, right_index=True)
        display(batch_info)

    result = pd.DataFrame(rows, index = index)

    return result

def cell_populations(adata : sc.AnnData, metric = 'percent'):
    cell_populations = adata.obs[['cell_type', 'batch', 'site']].groupby(['cell_type', 'batch']).count(). \
        rename(columns = {'site' : 'count'}).reset_index(). \
        pivot(index = 'cell_type', columns = 'batch')

    cell_populations.columns = cell_populations.columns.droplevel()

    if metric == 'percent':
        cell_populations /= cell_populations.sum(axis = 0)
        with pd.option_context("precision", 2):
            display(cell_populations * 100)
    else:
        raise ValueError(f'Unknown metric {metric}')

def plot_batch_counts(adata):
    batches = adata.obs['batch'].unique()
    fg = crna.pl.figure_grid(ncol = 4, ntotal = len(batches), figsize = (15, 5))
    for batch in batches:
        ax = next(fg)
        batch_data = adata.obs.loc[adata.obs['batch'] == batch, :]
        ax.set_title(batch)
        ax.set_xlim([0, 20000]); ax.set_ylim([0, 600])
        sb.histplot(batch_data['total_counts'], ax = ax)

def select_batches(data, batches):
    if isinstance(batches, str):
        batches = [batches]
    if isinstance(data, sc.AnnData):
        result = data[data.obs['batch'].isin(batches), :]
        return result
    else:
        result = {}
        for k, v in data.items():
            result[k] = v[v.obs['batch'].isin(batches), :]
        return result


def compare_nearest_neighbours(adata1 : sc.AnnData, adata2 : sc.AnnData):
    pass