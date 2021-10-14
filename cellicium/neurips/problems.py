import numpy as np
import scanpy as sc
import typing as tp
import sklearn.neighbors as sknn
import sklearn.preprocessing as skpp
import scipy.sparse as sparse
from cellicium.logging import logger as log
from .base import ProblemDataset
from .siamese import SiameseModelManager

def compute_pairing_matrix_neighbours(z1_data : sc.AnnData, z2_data : sc.AnnData, n_neighbors = 10) -> sc.AnnData:
    '''
    :param z1_data:
    :param z2_data:
    :param n_neighbors:
    :return:
        X: The sparse pairing matrix. A value of 1 in this matrix
        means this modality 1 profile (row) corresponds to a modality 2 profile (column)
    '''
    Z1 = z1_data.X
    Z2 = z2_data.X
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
    neighbors = sknn.NearestNeighbors(n_neighbors = n_neighbors).fit(Z1)
    distances, indices = neighbors.kneighbors(X = Z2)
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

    pairing_matrix = sparse.csr_matrix(
        (ind_values, (ind_i, ind_j)),
        shape = (Z1.shape[0], Z2.shape[0])
    )

    # Normalize values to sum to 1
    pairing_matrix = skpp.normalize(pairing_matrix, norm="l1", axis=1)

    return pairing_matrix

def score_pairing_matrix(predicted, solution):
    X_pred = predicted.X if isinstance(predicted, sc.AnnData) else predicted
    X_pred = X_pred.copy()
    X_solution = solution.X if isinstance(solution, sc.AnnData) else solution
    # Remove negatives
    X_pred.data[X_pred.data < 0] = 0
    row_sums = np.asarray(X_pred.sum(axis = 1)).flatten()
    row_norm = sparse.diags(1. / row_sums).tocsr()
    # Normalize rows
    X_pred = row_norm.multiply(X_pred)
    match_score = X_pred.multiply(X_solution)
    match_score = np.sum(match_score) / X_pred.shape[0]
    return match_score

def score_match(ds : ProblemDataset, mm : SiameseModelManager, n_neighbors = 10):
    z1_data, z2_data = mm.transform_to_common_space([ds.train_mod1, ds.train_mod2])
    PM = compute_pairing_matrix_neighbours(z1_data, z2_data, n_neighbors = n_neighbors)
    match_score = score_pairing_matrix(PM, ds.train_sol)
    print(f'Training accuracy match score: {match_score:.7f}')
    if ds.test_sol is not None:
        z1_data, z2_data = mm.transform_to_common_space([ds.test_mod1, ds.test_mod2])
        PM = compute_pairing_matrix_neighbours(z1_data, z2_data, n_neighbors = n_neighbors)
        match_score = score_pairing_matrix(PM, ds.test_sol)
        print(f'Testing accuracy match score: {match_score:.7f}')

