import numpy as np
import pandas as pd
import scanpy as sc
import os

from cellicium.logging import logger as log
import cellicium.scrna as crna
from cellicium.utils import display
import seaborn as sb
import scipy.sparse as sparse
from .base import ProblemDataset

base_path_match_modality = '/home/jovyan/neurips/starter-kits/match-modality/output/datasets/match_modality'

base_paths = {
    'benchmark-cite': '/home/jovyan/neurips/datasets/cite/',
    'benchmark-multiome': '/home/jovyan/neurips/datasets/multiome/',
    'openproblems_bmmc_cite_phase1_mod2' : base_path_match_modality,
    'openproblems_bmmc_cite_phase1_rna': base_path_match_modality,
    'openproblems_bmmc_multiome_phase1_mod2': base_path_match_modality,
    'openproblems_bmmc_multiome_phase1_rna': base_path_match_modality
}

def load_data(name, **kwargs) -> ProblemDataset:
    rng = np.random.default_rng(seed = 42)
    if name == 'benchmark-multiome':
        base_path = base_paths[name]
        gex_path = os.path.join(base_path, 'multiome_gex_processed_training.h5ad')
        atac_path = os.path.join(base_path, 'multiome_atac_processed_training.h5ad')
        # log.info(os.path.abspath(gex_path))
        log.info(f'Loading {gex_path}')
        gex_data = sc.read_h5ad(gex_path)
        log.info(f'Loading {atac_path}')
        atac_data = sc.read_h5ad(atac_path)
        # No total_counts is present in gex_data
        gex_data.obs['total_counts'] = np.sum(gex_data.layers['counts'], axis = 1)
        gex_data.obs['site'] = gex_data.obs['batch'].str.slice(0, 2)
        gex_data.obs['donor'] = gex_data.obs['batch'].str.slice(2, 4)
        atac_data.obs['site'] = atac_data.obs['batch'].str.slice(0, 2)
        atac_data.obs['donor'] = atac_data.obs['batch'].str.slice(2, 4)

        gex_data.uns['modality'] = 'GEX'
        atac_data.uns['modality'] = 'atac'

        all_batches = set(gex_data.obs['batch'].unique())
        test_batches = set(kwargs.pop('test_batches', ['s1d2', 's3d7']))
        train_batches = all_batches.difference(test_batches)
        log.info(f'Train batches: {train_batches}')
        log.info(f'Test batches: {test_batches}')
        train_selector = gex_data.obs['batch'].isin(train_batches)
        test_selector = gex_data.obs['batch'].isin(test_batches)
        n_train = np.sum(train_selector)
        n_test = np.sum(test_selector)
        log.info(f'Num train samples: {n_train}')
        log.info(f'Num test samples: {n_test}')

        train_mix = rng.choice(n_train, size = n_train, replace = False)
        test_mix = rng.choice(n_test, size = n_test, replace = False)

        result = ProblemDataset(
            train_mod1 = gex_data[train_selector, :][train_mix, :],
            train_mod2 = atac_data[train_selector, :],
            train_sol= sc.AnnData(X = sparse.diags(np.ones(n_train)).tocsr())[train_mix, :],
            test_mod1 = gex_data[test_selector, :][test_mix, :],
            test_mod2= atac_data[test_selector, :],
            test_sol= sc.AnnData(X = sparse.diags(np.ones(n_test)).tocsr())[test_mix, :],
            modality1 = 'GEX',
            modality2 = 'ATAC'
        )
    elif name == 'benchmark-cite':
        base_path = base_paths[name]
        gex_path = os.path.join(base_path, 'cite_gex_processed_training.h5ad')
        atac_path = os.path.join(base_path, 'cite_adt_processed_training.h5ad')
        # log.info(os.path.abspath(gex_path))
        log.info(f'Loading {gex_path}')
        gex_data = sc.read_h5ad(gex_path)
        log.info(f'Loading {atac_path}')
        adt_data = sc.read_h5ad(atac_path)
        # No total_counts is present in gex_data
        gex_data.obs['total_counts'] = np.sum(gex_data.layers['counts'], axis = 1)
        gex_data.obs['site'] = gex_data.obs['batch'].str.slice(0, 2)
        gex_data.obs['donor'] = gex_data.obs['batch'].str.slice(2, 4)
        adt_data.obs['site'] = adt_data.obs['batch'].str.slice(0, 2)
        adt_data.obs['donor'] = adt_data.obs['batch'].str.slice(2, 4)

        gex_data.uns['modality'] = 'GEX'
        adt_data.uns['modality'] = 'ADT'

        all_batches = set(gex_data.obs['batch'].unique())
        test_batches = set(kwargs.pop('test_batches', ['s1d2', 's3d7']))
        train_batches = all_batches.difference(test_batches)
        log.info(f'Train batches: {train_batches}')
        log.info(f'Test batches: {test_batches}')
        train_selector = gex_data.obs['batch'].isin(train_batches)
        test_selector = gex_data.obs['batch'].isin(test_batches)
        n_train = np.sum(train_selector)
        n_test = np.sum(test_selector)
        log.info(f'Num train samples: {n_train}')
        log.info(f'Num test samples: {n_test}')

        train_mix = rng.choice(n_train, size = n_train, replace = False)
        test_mix = rng.choice(n_test, size = n_test, replace = False)

        result = ProblemDataset(
            train_mod1 = gex_data[train_selector, :][train_mix, :],
            train_mod2 = adt_data[train_selector, :],
            train_sol= sc.AnnData(X = sparse.diags(np.ones(n_train)).tocsr())[train_mix, :],
            test_mod1 = gex_data[test_selector, :][test_mix, :],
            test_mod2= adt_data[test_selector, :],
            test_sol= sc.AnnData(X = sparse.diags(np.ones(n_test)).tocsr())[test_mix, :],
            modality1 = 'GEX',
            modality2 = 'ADT'
        )
    elif name in {'openproblems_bmmc_cite_phase1_mod2', 'openproblems_bmmc_cite_phase1_rna', 'openproblems_bmmc_multiome_phase1_mod2', 'openproblems_bmmc_multiome_phase1_rna'}:
        base_path = base_paths[name]
        input_train_mod1 = sc.read_h5ad(os.path.join(base_path, f'{name}/{name}.censor_dataset.output_train_mod1.h5ad'))
        input_train_mod2 = sc.read_h5ad(os.path.join(base_path, f'{name}/{name}.censor_dataset.output_train_mod2.h5ad'))
        input_train_sol = sc.read_h5ad(os.path.join(base_path, f'{name}/{name}.censor_dataset.output_train_sol.h5ad'))
        input_test_mod1 = sc.read_h5ad(os.path.join(base_path, f'{name}/{name}.censor_dataset.output_test_mod1.h5ad'))
        input_test_mod2 = sc.read_h5ad(os.path.join(base_path, f'{name}/{name}.censor_dataset.output_test_mod2.h5ad'))
        input_test_sol = sc.read_h5ad(os.path.join(base_path, f'{name}/{name}.censor_dataset.output_test_sol.h5ad'))
        modality1 = input_train_mod1.var['feature_types'][0]
        modality2 = input_train_mod2.var['feature_types'][0]

        input_train_mod1.uns['modality'] = modality1
        input_train_mod2.uns['modality'] = modality2
        input_test_mod1.uns['modality'] = modality1
        input_test_mod2.uns['modality'] = modality2

        train_batches = set(input_train_mod1.obs['batch'].unique())
        test_batches = set(input_test_mod1.obs['batch'].unique())
        log.info(f'Train batches: {train_batches}')
        log.info(f'Test batches: {test_batches}')
        log.info(f'Num train samples: {input_train_mod1.obs.shape[0]}')
        log.info(f'Num test samples: {input_test_mod1.obs.shape[0]}')


        result = ProblemDataset(
            train_mod1 = input_train_mod1,
            train_mod2 = input_train_mod2,
            train_sol= input_train_sol,
            test_mod1 = input_test_mod1,
            test_mod2= input_test_mod2,
            test_sol= input_test_sol,
            modality1 = modality1,
            modality2 = modality2
        )
    elif name == 'run':
        input_train_mod1 = sc.read_h5ad(kwargs['input_train_mod1'])
        input_train_mod2 = sc.read_h5ad(kwargs['input_train_mod2'])
        input_train_sol = sc.read_h5ad(kwargs['input_train_sol'])
        input_test_mod1 = sc.read_h5ad(kwargs['input_test_mod1'])
        input_test_mod2 = sc.read_h5ad(kwargs['input_test_mod2'])
        modality1 = input_train_mod1.var['feature_types'][0]
        modality2 = input_train_mod2.var['feature_types'][0]

        input_train_mod1.uns['modality'] = modality1
        input_train_mod2.uns['modality'] = modality2
        input_test_mod1.uns['modality'] = modality1
        input_test_mod2.uns['modality'] = modality2

        result = ProblemDataset(
            train_mod1 = input_train_mod1,
            train_mod2 = input_train_mod2,
            train_sol= input_train_sol,
            test_mod1 = input_test_mod1,
            test_mod2= input_test_mod2,
            test_sol= None,
            modality1 = modality1,
            modality2 = modality2
        )


    else:
        raise ValueError(f'Unknown dataset {name}, available data sets are: {base_paths.keys()}')

    return result

def load_nmf_spectra(spectra_file):
    nmf_spectra = pd.read_csv(spectra_file, sep = '\t', index_col = 0)
    nmf_spectra.index = nmf_spectra.index.map(str)
    return nmf_spectra

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
    fg = crna.pl.figure_grid(n_col= 4, n_total= len(batches), figsize = (15, 5))
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

