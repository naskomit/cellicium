import numpy as np
import pandas as pd
import scanpy as sc
from cellicium.logging import logger as log
import sklearn.decomposition as skdcmp
from .base import ProblemDataset
import cellicium.cnmf as cnmf


def nmf_activity(adata : sc.AnnData, nmf_spectra, normalize = True):
    genes = nmf_spectra.columns.values
    norm_counts = adata[:, genes].copy()

    sc.pp.scale(norm_counts, zero_center=False)
    if np.isnan(norm_counts.X.data).sum() > 0:
        print('Warning NaNs in normalized counts matrix')

    nmf_default_kwargs = {
        'alpha': 0.0,
        'beta_loss': 'frobenius',
        'init': 'random',
        'l1_ratio': 0.0,
        'max_iter': 1000,
        'regularization': None,
        'solver': 'cd',
        'tol': 0.0001,
        'n_components': nmf_spectra.shape[0],
        'H' : nmf_spectra.values.astype('float32'),
        'update_H': False
    }

    nmf_kwargs = nmf_default_kwargs

    log.info('Factorizing matrix ...')
    activities, _, _ = skdcmp.non_negative_factorization(norm_counts.X, **nmf_kwargs)
    log.info('Done factorizing matrix!')
    total_activity = np.sum(activities, axis = 1)
    if normalize:
        activities = activities / total_activity.reshape((-1, 1))
    activities = pd.DataFrame(activities, index = adata.obs.index, columns = nmf_spectra.index)
    result = sc.AnnData(X = activities, obs = adata.obs)
    result.obs['total_activity'] = total_activity
    if 'modality' in adata.uns:
        result.uns['modality'] = adata.uns['modality'] + '_NMF'
    return result

def nmf_transform(ds : ProblemDataset, spectral_map, layer = None):
    result = {}
    for field in ['train_mod1', 'train_mod2', 'test_mod1', 'test_mod2']:
        adata = getattr(ds, field)
        modality = adata.uns['modality']
        spectra = spectral_map[modality]
        if layer is not None:
            nmf_input = sc.AnnData(X = adata.layers[layer], obs = adata.obs,
                                   var = adata.var, uns = {'modality': modality})
        else:
            nmf_input = adata
        result[field] = nmf_activity(nmf_input, spectra)
        # if 'modality' not in result[field].uns:
        #     result[field].uns['modality'] = adata.uns['modality'] + '_NMF'
    result['modality1'] = ds.modality1
    result['modality2'] = ds.modality2
    result['train_sol'] = ds.train_sol
    result['test_sol'] = ds.test_sol
    return ProblemDataset(**result)


def get_cnmf_runner(adata, workdir, name, run_analysis = False, use_counts = True,
                    num_components = 50, num_iterations = 100, minibatch_size = None, num_workers = 4):
    cnmf_runner = cnmf.CNMFRunner(workdir, name, num_workers = num_workers)
    if run_analysis:
        log.info('Saving count matrix')
        # adata_count = sc.AnnData(X = adata.layers['counts'], var = adata.var, obs = adata.obs)
        log.info('Preparing ...')
        cnmf_runner.prepare(adata, use_counts = use_counts, num_components = num_components, num_iterations = num_iterations)
        log.info('Factorizing ...')
        cnmf_runner.factorize(minibatch_size = minibatch_size)
        log.info('Combining ...')
        cnmf_runner.combine()
        log.info('Creating consensus processes ...')
        cnmf_runner.consensus(density_threshold = 0.5)
    else:
        cnmf_runner.set_current_consensus(num_components = num_components)
    return cnmf_runner