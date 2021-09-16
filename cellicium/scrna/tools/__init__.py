import numpy as np
import scipy.optimize as spopt
from .rotation import fit_matrix, fit_multilayer
from .smoothing import bin_smooth, low_pass_filter


def add_intron_data(adata, intron_data):
  assert(adata.n_obs == intron_data.n_obs, "Number of observations must be the same for both exonic and intronic reads")
  #exon_genes = exon
  intron_layer = np.zeros(adata.X.shape)
  for i, v in enumerate(adata.var.index):
      #print(i, v)
      if v in intron_data.var.index:
          intron_layer[:, i] = intron_data[:, v].X.flatten()
  adata.layers['spliced'] = adata.X
  adata.layers['unspliced'] = intron_layer
  return adata


def fit_peak(data):
    n_points = data.shape[0]
    def f1(x):
        mu = x[0]
        sigma = x[1]
        A = x[2]
        B = x[3]
        z = np.linspace(0, 1, n_points)
        y = A * np.exp(-((z - mu) / sigma)**2) + B
        return y

    def f1_loss(data_in):
        return lambda x: np.linalg.norm(f1(x) - data_in)

    res = spopt.minimize(
        f1_loss(data), x0 = [0.5, 0.5, 2, 0],
        bounds = [[0, 1], [0.1, 1], [0, 10], [0, 10]])
    return (f1(res.x), res)
