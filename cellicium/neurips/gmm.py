import numpy as np
from scipy import stats

# import tensorflow as tf
# import tensorflow.keras as tfk
from tqdm import tqdm
from cellicium.logging import logger as log

from sklearn.base import BaseEstimator, DensityMixin
from sklearn.mixture import GaussianMixture
import scanpy as sc

def _log_norm(x, scale_factor=10000):
    x = x.astype('float32')
    x_sum = np.sum(x)
    return np.log1p(x / (x_sum + np.finfo(x.dtype).eps) * scale_factor)

class _DummyGMM:
    def __init__(self):
        self.means_ = None
        self.precisions_ = None

    def fit(self, X, y=None):
        self.means_ = np.array([np.mean(X)])
        self.precisions_ = np.array([1. / np.var(X)])

    def predict(self, X):
        return (X >= self.means_[0]).astype(np.float32)

    def predict_proba(self, X):
        return self.predict(X)

    def predict_z_score(self, X):
        raise ValueError('Unimplemented')
        #return (X - self.means_[0]) * np.sqrt(self.precisions_[0])

class ProbabilisticEmbedding(BaseEstimator, DensityMixin):
    def __init__(self,
                 n_components_per_class : int = 2,
                 positive_component : int = 1,
                 log_norm : bool = True,
                 clip_quartile : float = 0.,
                 remove_zeros : bool = True,
                 ci_threshold : float = -0.68,
                 random_state : int = 8,
                 verbose : bool = True):
        super().__init__()
        self.n_components_per_class = n_components_per_class
        self.positive_component = positive_component
        self.log_norm =log_norm
        self.clip_quartile = clip_quartile
        self.remove_zeros = remove_zeros
        self.ci_threshold = ci_threshold
        self.random_state = random_state
        self.verbose = verbose
        self._models = []

    @property
    def n_classes(self):
        return len(self._models)

    def normalize(self, x, test_mode=False):
        if x.ndim > 1:
            x = x.ravel()
        n_samples = len(x)
        assert np.all(x >= 0), "Only support non-negative values"
        if self.remove_zeros and not test_mode:
            x = x[x > 0]
            if len(x) != n_samples:
                x = np.concatenate([[0], x], axis=0)
        # if self.clip_quartile > 0:
        #     x = _clipping_quartile(x, alpha=self.clip_quartile, test_mode=test_mode)
        if self.log_norm:
            x = _log_norm(x)
        return x

    def fit(self, X):
        n_classes = X.shape[1]
        it = tqdm(list(range(n_classes))) #if self.verbose else range(n_classes)
        for i in it:
            x_train = self.normalize(X[:, i], test_mode = False)
            x_train = x_train[:, np.newaxis]
            try:
                gmm = GaussianMixture(
                    n_components = self.n_components_per_class,
                    covariance_type = 'diag',
                    init_params = 'kmeans',
                    n_init = 8,
                    max_iter = 120,
                    random_state = self.random_state
                )
                gmm.fit(x_train)
            except ValueError as e:
                if "ill-defined empirical covariance" in str(e):
                    log.warn(f'Ill-defined empirical covariance for component {i}, revreting to using dummy model')
                    gmm = _DummyGMM()
                    gmm.fit(x_train)
                else:
                    import traceback
                    traceback.print_exc()
                    raise e

            means_ = gmm.means_.ravel()
            # print(means_)
            order = np.argsort(means_)
            self._models.append((order, gmm))

        it.close()

    def fit_transform(self, X, out = 'binary'):
        self.fit(X)
        if out == 'binary':
            return self.predict(X)
        elif out == 'probability':
            return self.predict_probability(X)
        elif out == 'z-score':
            return self.z_score(X)
        else:
            raise ValueError("`out` must be one of 'binary', 'probability' or 'z-score'")

    def _predict(self, X, threshold):
        assert X.shape[1] == self.n_classes, "Number of classes mis-match"
        y = []
        for i, (order, gmm) in enumerate(self._models):
            x_test = self.normalize(X[:, i], test_mode=True)
            # binary thresholding
            if isinstance(gmm, _DummyGMM):
                x_out = gmm.predict(x_test)
            elif threshold is not None:
                ci = stats.norm.interval(
                    np.abs(threshold),
                    loc = gmm.means_[order[self.positive_component]],
                    scale = np.sqrt(1 / gmm.precisions_[order[self.positive_component]]))
                x_out = (x_test >=
                         (ci[0] if threshold < 0 else ci[1])).astype('float32')
            # probabilizing
            else:
                probas = gmm.predict_proba(
                    x_test[:, np.newaxis]).T[order][self.positive_component:]
                x_out = np.mean(probas, axis=0)
            x_out = x_out[:, np.newaxis]
            y.append(x_out)
        return np.concatenate(y, axis=1)

    def z_score(self, X):
        assert X.shape[1] == self.n_classes, "Number of classes mis-match"
        y = []
        for i, (order, gmm) in enumerate(self._models):
            x_test = self.normalize(X[:, i], test_mode=True)
            # binary thresholding
            if isinstance(gmm, _DummyGMM):
                x_out = gmm.predict_z_score(x_test)
            else:
                #mean = gmm.means_[order[self.positive_component]]
                scale_inv = np.sqrt(gmm.precisions_[order[self.positive_component]])
                x_out = x_test * scale_inv
            x_out = x_out[:, np.newaxis]
            y.append(x_out)
        return np.concatenate(y, axis=1)

    def predict(self, X):
        return self._predict(X, threshold = self.ci_threshold)

    def predict_probability(self, X):
        return self._predict(X, threshold = None)


def gmm_embed(adata, batch_var = 'batch', out = 'binary'):
    batches = adata.obs[batch_var].unique()
    embedding = sc.AnnData(X = np.zeros((adata.n_obs, adata.n_vars)), obs = adata.obs, var = adata.var)
    for batch in batches:
        log.info(f'Processing batch {batch}')
        batch_selector = (adata.obs[batch_var] == batch)
        adata_batch = adata[batch_selector, :]
        X = np.asarray(adata_batch.layers['counts'].todense())
        pemb = ProbabilisticEmbedding()
        probs = pemb.fit_transform(X, out = out)
        embedding[batch_selector, :].X = probs
    return embedding