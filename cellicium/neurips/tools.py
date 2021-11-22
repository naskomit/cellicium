import numpy as np
import scipy.sparse as sparse
import pandas as pd
import scanpy as sc
import sklearn.decomposition as skdcmp
import sklearn.neighbors as sknn
import sklearn.preprocessing as skpp
import cellicium.cnmf as cnmf
from cellicium.logging import logger as log
from .base import ProblemDataset, ModelManagerBase
import cellicium.scrna as crna
import seaborn as sb
import matplotlib.pyplot as plt
import typing as tp

###############################
# NMF tools
###############################

def nmf_activity(adata : sc.AnnData, nmf_spectra, normalize = True):
    genes = nmf_spectra.columns.values
    genes = genes[np.isin(genes, adata.var.index.values)]
    log.info(f'NMF using {genes.shape[0]} variables')
    norm_counts = adata[:, genes].copy()
    nmf_spectra = nmf_spectra.loc[:, genes]

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


##############################
## Batch correction tools
##############################
def compute_linear_batch_correction(adata : sc.AnnData, by : str = 'batch'):
    log.info('Computing batch coefficients...')
    import statsmodels.formula.api as smf
    var_columns = adata.var.index.values
    X = pd.DataFrame(adata.X.todense(), index = adata.obs.index, columns = adata.var.index)
    batch = pd.Categorical(adata.obs[by])
    X['_batch'] = batch

    # Compute batch coefficients
    batch_coefficients = np.zeros((batch.categories.shape[0], adata.n_vars))
    for i, var_column in enumerate(var_columns):
        mod = smf.glm(formula = f'Q("{var_column}") ~ _batch ', data = X)
        res = mod.fit()
        # print(res.params[:])
        batch_coefficients [1:, i] = res.params[1:]

    batch_coefficients = pd.DataFrame(
        data = batch_coefficients, columns = var_columns, index = batch.categories)

    return batch_coefficients


def apply_linear_batch_correction(adata : sc.AnnData, batch_coefficients, by : str = 'batch'):
    # Correct batches
    var_columns = adata.var.index.values
    dX = pd.DataFrame({'_batch': adata.obs[by]}).join(
        batch_coefficients, on = '_batch', how = 'left')[var_columns]

    X = adata.X
    dX = dX[var_columns].values
    X = X - dX

    result = sc.AnnData(X = X, obs = adata.obs, var = adata.var)
    if 'modality' in adata.uns:
        result.uns['modality'] = adata.uns['modality']

    return result

##############################
## Problem tools
##############################
def normalize_gex_nmf(adata):
    adata = adata.copy()
    adata.X = np.log(1 + adata.X)
    return adata

def preprocess_adt(adata : sc.AnnData, annotations : pd.DataFrame) -> sc.AnnData:
    # hv_adt = adata.var.join(annotations[['double_peak']])['double_peak']
    # print(np.sum(hv_adt))
    # return adata[:, hv_adt]
    return adata

def prepare_training_data(input_data : ProblemDataset, nmf_spectra, adt_annotations : pd.DataFrame):
    #batch_var = 'site'
    if (input_data.modality1 == 'GEX') and (input_data.modality2 == 'ADT'):
        mod_combination = 'GEX+ADT'
        # adt_all_data = input_data.combined_mod2
        # batch_coeff_adt = compute_linear_batch_correction(adt_all_data, by = batch_var)
        input_data_nmf = nmf_transform(input_data, nmf_spectra, layer = 'counts')

        result = input_data._replace(
            train_mod1 = normalize_gex_nmf(input_data_nmf.train_mod1),
            test_mod1 = normalize_gex_nmf(input_data_nmf.test_mod1),
            train_mod2 = preprocess_adt(input_data.train_mod2, adt_annotations),
            test_mod2 = preprocess_adt(input_data.test_mod2, adt_annotations),
            modality1 = 'GEX_NMF')
        # train_mod2 = apply_linear_batch_correction(input_data.train_mod2, batch_coeff_adt, by = batch_var),
        # test_mod2 = apply_linear_batch_correction(input_data.test_mod2, batch_coeff_adt, by = batch_var),

    elif (input_data.modality1 == 'ADT') and (input_data.modality2 == 'GEX'):
        mod_combination = 'GEX+ADT'
        # adt_all_data = input_data.combined_mod1
        input_data_nmf = nmf_transform(input_data, nmf_spectra, layer = 'counts')

        result = input_data._replace(
            train_mod1 =  preprocess_adt(input_data.train_mod1, adt_annotations),
            test_mod1 = preprocess_adt(input_data.test_mod1, adt_annotations),
            train_mod2 = normalize_gex_nmf(input_data_nmf.train_mod2),
            test_mod2 = normalize_gex_nmf(input_data_nmf.test_mod2),
            modality2 = 'GEX_NMF')

    elif (input_data.modality1 == 'GEX') and (input_data.modality2 == 'ATAC'):
        mod_combination = 'GEX+ATAC'
        input_data_nmf = nmf_transform(input_data, nmf_spectra, layer = 'counts')

        result = input_data._replace(
            train_mod1 =  normalize_gex_nmf(input_data_nmf.train_mod1),
            test_mod1 = normalize_gex_nmf(input_data_nmf.test_mod1),
            train_mod2 = normalize_gex_nmf(input_data_nmf.train_mod2),
            test_mod2 = normalize_gex_nmf(input_data_nmf.test_mod2),
            modality1 = 'GEX_NMF',
            modality2 = 'ATAC_NMF')

    elif (input_data.modality1 == 'ATAC') and (input_data.modality2 == 'GEX'):
        mod_combination = 'GEX+ATAC'
        input_data_nmf = nmf_transform(input_data, nmf_spectra, layer = 'counts')

        result = input_data._replace(
            train_mod1 =  normalize_gex_nmf(input_data_nmf.train_mod1),
            test_mod1 = normalize_gex_nmf(input_data_nmf.test_mod1),
            train_mod2 = normalize_gex_nmf(input_data_nmf.train_mod2),
            test_mod2 = normalize_gex_nmf(input_data_nmf.test_mod2),
            modality1 = 'ATAC_NMF',
            modality2 = 'GEX_NMF')

    else:
        mod_combination = None
        result = None


    return result



def compute_pairing_matrix_neighbours(z1_data : sc.AnnData, z2_data : sc.AnnData, n_neighbors = 10):
    '''
    :param z1_data:
    :param z2_data:
    :param n_neighbors:
    :return:
        X: The sparse pairing matrix. A value of 1 in this matrix
        means this modality 1 profile (row) corresponds to a modality 2 profile (column)
    '''
    Z1 = z1_data.X if isinstance(z1_data, sc.AnnData) else z1_data
    Z2 = z2_data.X if isinstance(z2_data, sc.AnnData) else z2_data
    n_obs = Z1.shape[0]
    # # Cluster
    # log.info('Clustering ...')
    # import sklearn.cluster as clust
    # kmc = clust.KMeans(n_clusters = 50, random_state=0).fit(np.vstack([Z1, Z2]))
    # labels = kmc.labels_
    # centers = kmc.cluster_centers_
    # Z1 = Z1 - kmc.cluster_centers_[kmc.labels_[:n_obs]]
    # Z2 = Z2 - kmc.cluster_centers_[kmc.labels_[n_obs:]]
    # result = kmc
    # normalization
    # Z1 = Z1 / np.sqrt(np.sum(np.square(Z1), axis = 1)).reshape([-1, 1])
    # Z2 = Z2 / np.sqrt(np.sum(np.square(Z2), axis = 1)).reshape([-1, 1])
    log.info('Computing neighbours ...')
    neighbors = sknn.NearestNeighbors(n_neighbors = n_neighbors, p = 1).fit(Z2)
    distances, indices = neighbors.kneighbors(X = Z1)
    log.info('Assembling result ...')
    ind_i = np.tile(np.arange(Z1.shape[0]), (n_neighbors, 1)).T.flatten()
    ind_j = indices.flatten()
    # ind_dist = distances.flatten()
    # ind_values = np.exp(-ind_dist ** 2)
    ind_values = np.zeros(Z1.shape[0] * n_neighbors)
    probs = [1.0]
    #n_wrong_batch = Z1.obs['batch'] == Z2indices[:, 0]
    for k in range(len(probs)):
        ind_values[k::n_neighbors] = probs[k]

    # print(ind_i)
    # print(ind_j)
    # print(ind_values)
    #
    pairing_matrix = sparse.csr_matrix(
        (ind_values, (ind_i, ind_j)),
        shape = (Z1.shape[0], Z2.shape[0])
    )

    # Normalize values to sum to 1
    pairing_matrix = skpp.normalize(pairing_matrix, norm = "l1", axis = 1)

    return pairing_matrix

def compute_pairing_matrix(z1_data : sc.AnnData, z2_data : sc.AnnData, by = None, **kwargs):
    n_neighbors = kwargs.pop('n_neighbours', 10)
    if by is None:
        pm = compute_pairing_matrix_neighbours(z1_data, z2_data, n_neighbors)
    else:
        pm = sparse.dok_matrix((z1_data.n_obs, z2_data.n_obs))
        batches = pd.Categorical(z1_data.obs[by])
        for batch in batches.categories:
            log.info(f'Batch {batch}')
            batch_selector_1 = (z1_data.obs[by] == batch)
            batch_selector_2 = (z2_data.obs[by] == batch)
            z1_batch = z1_data[batch_selector_1, :]
            z1_indexer = np.where(batch_selector_1)[0]
            z2_batch = z2_data[batch_selector_2, :]
            z2_indexer = np.where(batch_selector_2)[0]
            pm_batch = compute_pairing_matrix_neighbours(z1_batch, z2_batch, n_neighbors)
            for row, col, val in zip(*sparse.find(pm_batch)):
                pm[z1_indexer[row], z2_indexer[col]] = val
    return pm.tocsr()

def score_pairing_matrix(predicted, solution):
    X_pred = predicted.X if isinstance(predicted, sc.AnnData) else predicted
    X_pred = X_pred.copy()
    X_solution = solution.X if isinstance(solution, sc.AnnData) else solution
    # Remove negatives
    X_pred.data[X_pred.data < 0] = 0
    row_sums = np.asarray(X_pred.sum(axis = 1)).flatten()
    row_norm = sparse.diags(1. / row_sums).tocsr()
    # Normalize rows
    X_pred = row_norm.dot(X_pred)
    match_score = X_pred.multiply(X_solution)
    match_score = np.sum(match_score) / X_pred.shape[0]
    return match_score

def score_match(ds : ProblemDataset, mm : ModelManagerBase, by = None, **kwargs):
    n_train = ds.train_mod1.n_obs
    result = []
    if n_train > 0:
        z1_data, z2_data = mm.transform_to_common_space([ds.train_mod1, ds.train_mod2])
        PM = compute_pairing_matrix(z1_data, z2_data, by = by, **kwargs)
        train_match_score = score_pairing_matrix(PM, ds.train_sol)
        #print(f'Training accuracy match score: {1000 * train_match_score:.2f}%% for {n_train} samples')
        result.append({'group': 'train', 'by': by, 'score': train_match_score, 'n_samples': n_train})

    n_test = ds.test_mod1.n_obs
    if (ds.test_sol is not None) and (n_test > 0):
        z1_data, z2_data = mm.transform_to_common_space([ds.test_mod1, ds.test_mod2])
        PM = compute_pairing_matrix(z1_data, z2_data, by = by, **kwargs)
        test_match_score = score_pairing_matrix(PM, ds.test_sol)
        #print(f'Testing accuracy match score: {1000 * test_match_score:.2f}%% for {n_test} samples')
        result.append({'group': 'test', 'by': by, 'score': test_match_score, 'n_samples': n_test})

    return result


def score_predict(ds: ProblemDataset, modality2 : str, mm : ModelManagerBase):
    from sklearn.metrics import mean_squared_error
    n_train = ds.train_mod1.n_obs
    if n_train > 0:
        adata_pred = mm.predict_modality(ds.train_mod1)
        train_error = mean_squared_error(adata_pred.X, ds.train_mod2.X, squared = False)
        print(f'Training prediction error: {train_error:.7f} for {n_train} samples')
    else:
        train_error = None

    if (ds.test_mod2 is not None) and (ds.test_mod1.n_obs > 0):
        n_test = ds.test_mod1.n_obs
        adata_pred = mm.predict_modality(ds.test_mod1)
        test_error = mean_squared_error(adata_pred.X, ds.test_mod2.X, squared = False)
        print(f'Testing prediction error: {test_error:.7f} for {n_test} samples')
    else:
        test_error = None
    return train_error, test_error


def plot_spectra(spectra : pd.DataFrame):
    result = {}
    n_spectra = spectra.shape[0]
    for prog in range(n_spectra):
        spectrum = spectra.iloc[prog, :]
        top_genes = spectrum.sort_values(ascending = False)[:30]
        result[f'{prog}_genes'] = top_genes.index.values
        result[f'{prog}_coeff'] = top_genes.values
    result = pd.DataFrame(result)

    fg = crna.pl.figure_grid(n_row = 17, n_col = 3, row_height = 7)
    for prog in range(n_spectra):
        ax = next(fg)
        ax.set_title(prog)
        crna.pl.text_plot(result, x = f'{prog}_genes', y = f'{prog}_coeff', ax = ax)


def intersect_spectra(spectra_1 : pd.DataFrame, spectra_2 : pd.DataFrame,
                      n_top : int = 30, sort = False, **kwargs):
    top_genes_1 = []
    top_genes_2 = []
    n1 = spectra_1.shape[0]
    n2 = spectra_2.shape[0]
    for i in range(n1):
        spectrum = spectra_1.iloc[i, :]
        top_genes = spectrum.sort_values(ascending = False)[:n_top]
        top_genes_1.append(set(top_genes.index.values))
    for i in range(n2):
        spectrum = spectra_2.iloc[i, :]
        top_genes = spectrum.sort_values(ascending = False)[:n_top]
        top_genes_2.append(set(top_genes.index.values))
    # print(top_genes_1)
    # print(top_genes_2)
    intersection_map = np.zeros((n1, n2))
    for i1 in range(n1):
        for i2 in range(n2):
            intersection_map[i1, i2] = len(top_genes_1[i1].intersection(top_genes_2[i2]))

    if sort:
        row_ord = np.argsort(np.argmax(intersection_map, axis=0))
        intersection_map = intersection_map[:, row_ord]
        xticklabels = np.arange(n2)[row_ord]
    else:
        xticklabels = 'auto'

    fg = plt.figure(figsize = (20, 20))
    ax = fg.gca()
    label1 = kwargs.pop('label1', None)
    label2 = kwargs.pop('label2', None)
    sb.heatmap(intersection_map, ax = ax, cmap = plt.cm.jet,
               xticklabels = xticklabels, **kwargs)
    if label1 is not None:
        ax.set_ylabel(label1)
    if label2 is not None:
        ax.set_xlabel(label2)

    return intersection_map


def match_by_nearest_neighbours(ds : ProblemDataset, p_metric = 1, n_neighbors = 10):
    def ensure_dense(X):
        X_dense = X.todense() if sparse.issparse(X) else X
        X_dense = np.asarray(X_dense)
        return X_dense

    major_modality = ds.modality1
    minor_modality = ds.modality2
    X1_train = ensure_dense(ds.get_data('train', major_modality, sort = True).X)
    X2_train = ensure_dense(ds.get_data('train', minor_modality, sort = True).X)
    X1_test = ensure_dense(ds.get_data('test', major_modality).X)
    X2_test = ensure_dense(ds.get_data('test', minor_modality).X)

    nn_model_1 = sknn.NearestNeighbors(n_neighbors = n_neighbors, p = p_metric, n_jobs = -1).fit(X1_train)
    nn_model_2 = sknn.NearestNeighbors(n_neighbors = n_neighbors, p = p_metric, n_jobs = -1).fit(X2_train)

    __, neighbors_train_1 = nn_model_1.kneighbors(X1_train)
    __, neighbors_train_2 = nn_model_2.kneighbors(X2_train)
    __, neighbors_test_1 = nn_model_1.kneighbors(X1_test)
    __, neighbors_test_2 = nn_model_2.kneighbors(X2_test)

    common_neighbors_train = np.zeros(neighbors_train_1.shape[0])
    random_neighbors_train = np.zeros(10 * neighbors_train_2.shape[0])
    for i in range(neighbors_train_1.shape[0]):
        common_neighbors_train[i] = np.sum(np.isin(neighbors_train_1[i][1:], neighbors_train_2[i][1:]))
        for j in range(10):
            random_neighb = np.random.randint(low = 0, high = neighbors_train_1.shape[0])
            random_neighbors_train[i + j] = np.sum(np.isin(neighbors_train_1[i][1:], neighbors_train_2[random_neighb][1:]))

    common_neighbors_test = np.zeros(neighbors_test_1.shape[0])
    random_neighbors_test = np.zeros(10 * neighbors_test_2.shape[0])
    for i in range(neighbors_test_1.shape[0]):
        common_neighbors_test[i] = np.sum(np.isin(neighbors_test_1[i], neighbors_test_2[i]))
        for j in range(10):
            random_neighb = np.random.randint(low = 0, high = neighbors_test_1.shape[0])
            random_neighbors_test[i + j] = np.sum(np.isin(neighbors_test_1[i], neighbors_test_2[random_neighb]))

    return {'common_neighbors_train': np.mean(common_neighbors_train) / n_neighbors,
            'common_neighbors_train_gain': np.mean(common_neighbors_train) / (1e-8 + np.mean(random_neighbors_train)),
            'common_neighbors_test': np.mean(common_neighbors_test) / n_neighbors,
            'common_neighbors_test_gain': np.mean(common_neighbors_test) / (1e-8 + np.mean(random_neighbors_test))}


def check_neighbors_across_modalities(ds : ProblemDataset, p_metric = 1, n_neighbors = 10):
    def ensure_dense(X):
        X_dense = X.todense() if sparse.issparse(X) else X
        X_dense = np.asarray(X_dense)
        return X_dense

    major_modality = ds.modality1
    minor_modality = ds.modality2
    X1_train = ensure_dense(ds.get_data('train', major_modality, sort = True).X)
    X2_train = ensure_dense(ds.get_data('train', minor_modality, sort = True).X)

    nn_model_1 = sknn.NearestNeighbors(n_neighbors = n_neighbors, p = p_metric, n_jobs = -1).fit(X1_train)
    nn_model_2 = sknn.NearestNeighbors(n_neighbors = n_neighbors, p = p_metric, n_jobs = -1).fit(X2_train)

    __, neighbors_train_1 = nn_model_1.kneighbors(X1_train)
    __, neighbors_train_2 = nn_model_2.kneighbors(X2_train)

    neighbors_train_1 = neighbors_train_1[:, 1:]
    neighbors_train_2 = neighbors_train_2[:, 1:]

    return neighbors_train_1, neighbors_train_2

def compute_pairwise_regression(adata1 : sc.AnnData, adata2 : sc.AnnData):
    pass