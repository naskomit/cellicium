import numpy as np
import pandas as pd
import scanpy as sc
import typing as tp
import matplotlib.pyplot as plt
from IPython.display import display
import sklearn.neighbors as sknn
import sklearn.preprocessing as skpp
import scipy.sparse as sparse
import cellicium.scrna as crna
import cellicium.neurips.siamese as siamese
import abc
from cellicium.logging import logger as log
import os

class MatchModalityProblem(abc.ABC):
    def __init__(self, work_dir):
        self.work_dir = work_dir

    def load_data_file(self, path):
        abs_path = os.path.join(self.work_dir, path)
        log.info(f'Loading file {abs_path}')
        return sc.read_h5ad(abs_path)

    def load_data(self):
        self.input_train_mod1 = self.load_data_file('output/datasets/match_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod1.h5ad')
        self.input_train_mod2 = self.load_data_file('output/datasets/match_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod2.h5ad')
        self.input_train_sol = self.load_data_file('output/datasets/match_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_sol.h5ad')
        self.input_test_mod1 = self.load_data_file('output/datasets/match_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod1.h5ad')
        self.input_test_mod2 = self.load_data_file('output/datasets/match_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod2.h5ad')
        self.input_test_sol = self.load_data_file('output/datasets/match_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_sol.h5ad')
        self.modality1 = self.input_test_mod1.var['feature_types'][0]
        self.modality2 = self.input_test_mod2.var['feature_types'][0]
        log.info(f'Modality 1: {self.modality1}')
        log.info(f'Modality 2: {self.modality2}')


    @abc.abstractmethod
    def find_encoder(self, modality):
        pass

    @property
    @abc.abstractmethod
    def obs_fields(self):
        pass

    def compute_pairing_matrix_neighbours(self, z1_data : sc.AnnData, z2_data : sc.AnnData, n_neighbors = 5) -> sc.AnnData:
        # X: The sparse pairing matrix. A value of 1 in this matrix means this modality 1 profile (row) corresponds to a modality 2 profile (column)
        Z1 = z1_data.X
        Z2 = z2_data.X
        log.info('Computing neighbours ...')
        neighbors = sknn.NearestNeighbors(n_neighbors = n_neighbors).fit(Z1)
        distances, indices = neighbors.kneighbors(X = Z2)
        log.info('Assembling result ...')
        ind_i = np.tile(np.arange(Z1.shape[0]), (n_neighbors, 1)).T.flatten()
        ind_j = indices.flatten()
        ind_dist = distances.flatten()
        print(ind_dist[:50])
        #ind_values = np.exp(-ind_dist ** 2)
        ind_values = np.zeros(Z1.shape[0] * n_neighbors)
        probs = [1.0]
        for k in range(len(probs)):
            ind_values[k::n_neighbors] = probs[k]

        pairing_matrix = sparse.csr_matrix(
            (ind_values, (ind_i, ind_j)),
            shape = (Z1.shape[0], Z2.shape[0])
        )

        # Normalize values to sum to 1
        pairing_matrix = skpp.normalize(pairing_matrix, norm="l1", axis=1)

        # r, c, v = sparse.find(pairing_matrix)
        # i1 = np.where(r < 5)
        # print(r[i1][:20])
        # print(c[i1][:20])
        # print(v[i1][:20])

        result = sc.AnnData(
            X = pairing_matrix,
            uns = {'dataset_id': 1, 'method_id': 'NMF'})
        return result

    def score_pairing_matrix(self, predicted, solution):
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
        print(f'Match score: {match_score:.7f}')
        return match_score



    def transform_to_common_space(self, adata_list : tp.List[sc.AnnData], unify = False):
        Z_list = []
        for adata in adata_list:
            X = adata.X.todense() if sparse.issparse(adata.X) else adata.X
            modality = adata.uns['modality']
            Z_predict = self.find_encoder(modality).predict(X)
            Z_list.append(Z_predict)

        if unify:
            obs_list = []
            for adata in adata_list:
                obs = adata.obs[self.obs_fields].copy()
                modality = adata.uns['modality']
                obs['modality'] = modality
                obs.index = obs.index + ('_' + modality)
                obs_list.append(obs)
            result = sc.AnnData(X = np.vstack(Z_list), obs = pd.concat(obs_list))
        else:
            result = []
            for i, adata in enumerate(adata_list):
                modality = adata.uns['modality']
                result.append(sc.AnnData(X = Z_list[i], obs = adata.obs,
                                         uns = {'modality': modality}))

        return result

    def plot_integration_umap(self, adata_list : tp.List[sc.AnnData]):
        adata = self.transform_to_common_space(adata_list, unify = True)
        log.info('Computing neighbours ...')
        sc.pp.neighbors(adata)
        log.info('Computing umap ...')
        sc.tl.umap(adata)
        fg = crna.pl.figure_grid(ncol = 2, nrow = 1, figsize = (40, 20))
        log.info('Creating plot ...')
        sc.pl.umap(adata, color = 'modality', ax = next(fg), s = 20, show = False)
        sc.pl.umap(adata, color = 'cell_type', ax = next(fg), s = 20, show = False, legend_loc = 'on data')

class MatchModalitySolution_NMF_SNN(MatchModalityProblem):
    def __init__(self, smm : siamese.SiameseModelManager = None, **kwargs):
        self.method_id = 'NMF_SNN'
        self.smm = smm
        super().__init__(**kwargs)

    def find_encoder(self, modality):
        if modality in self.smm.modality_encoders:
            return self.smm.modality_encoders[modality]
        else:
            raise ValueError(f'No encoder found for modality {modality}')

    @property
    def obs_fields(self):
        return ['cell_type', 'batch', 'site', 'donor']

