import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..scrna.plotting import figure_grid

class TransformedVariable(tf.Module):
    def __init__(self, *args, **kwargs):
        self.transformation = kwargs.pop('transformation')
        self.var = tf.Variable(*args, **kwargs)

    def __call__(self):
        return self.transformation(self.var)

    def transform(self, x):
        return self.transformation(x)

    def numpy(self):
        return self.__call__().numpy()

class GeneSwitchModel(tf.Module):
    def __init__(self, t_data, y_data, name = None):
        super().__init__(name = name)
        self.t_data = tf.convert_to_tensor(t_data.reshape((-1, 1)))
        self.y_data = tf.convert_to_tensor(y_data)
        self.n_obs, self.n_var = y_data.shape
        sampling_init = 0.05
        transf_0_1 = lambda x: (tf.math.tanh(x) + 1) / 2
        def transf_0_p(p : float):
            return lambda x: p * (tf.math.tanh(x) + 1) / 2
        gene_means = np.mean(y_data, axis = 0)
        ampl_init =  gene_means / sampling_init
        initializer = lambda p: p * np.ones(self.n_var, dtype = np.float32)

        self.ampl = tf.Variable(ampl_init, shape = (self.n_var), name = 'ampl', constraint=lambda t: tf.clip_by_value(t, 1, 10000), dtype = np.float32)
        #self.offset = TransformedVariable(initializer(0.0), shape = (self.n_var), name = 'offset', transformation = tf.math.softplus, dtype = np.float32)
        self.mu = TransformedVariable(initializer(0.0), shape = (self.n_var), name = 'mu', dtype = np.float32, transformation = transf_0_1)
        self.sigma = TransformedVariable(initializer(0.0), shape = (self.n_var), name = 'sigma', dtype = np.float32, transformation = transf_0_p(0.5))
        self.sampling_fraction =  TransformedVariable(sampling_init, name = 'sampling_fraction', transformation = transf_0_1)

    def params_to_str(self):
        return f'sampling_fraction = {self.sampling_fraction.numpy():.3f}'
        # f'ampl= {self.ampl.numpy():.1f}, offset = {self.offset.numpy():.1f}, mu = {self.mu.numpy():.3f}, '\
        #         f'sigma = {self.sigma.numpy():.3f}, sampling_fraction = {self.sampling_fraction.numpy():.3f}'
    def make_sampling_dist(self, true_count):
        sampling_fraction = self.sampling_fraction()
        return tfp.distributions.NegativeBinomial(total_count = true_count, probs = sampling_fraction) #, validate_args = True)
        #return tfp.distributions.Binomial(total_count = true_count, probs = sampling_fraction)
#     def transform_vars(self, mu : tf.Tensor, sigma : tf.Tensor):
#         mu_1 = (tf.math.tanh(mu) + 1) / 2
#         sigma_1 = (tf.math.tanh(sigma) + 1) / 2
#         return mu_1, sigma_1

    def peak_shape(self, x : tf.Tensor, mu : tf.Tensor, sigma : tf.Tensor):
        return tf.math.exp(-tf.math.square(x - mu) / sigma)

    def __call__(self, t_data : tf.Tensor, training : bool = False):
        mu = self.mu()
        sigma = self.sigma()
        ampl = self.ampl
        x = t_data
        #self.offset() +
        true_count = ampl * (self.peak_shape(x - 1, mu, sigma) + self.peak_shape(x, mu, sigma) + self.peak_shape(x + 1, mu, sigma))
        #true_count = tf.round(true_count) #tf.cast(, dtype = tf.int32)
        dist = self.make_sampling_dist(true_count)
        return dist

    def compute_loss(self, dist : tfp.distributions.Distribution, y : tf.Tensor):
        #mse = tf.reduce_sum(tf.math.square(y_pred - y))
        # print(f'Dist shape {dist.log_prob(y).shape}')
        neg_log_prob = - tf.reduce_sum(dist.log_prob(y), axis = 0)
        loss = tf.reduce_mean(neg_log_prob)
        # print(f'Neg_log_prob {neg_log_prob.shape}')
        # kl = tf.square(self.sigma.var)
        # loss = neg_log_prob + 100 * kl
        return loss

    def train(self, n_iter = 1000, learning_rate = 0.3):
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        print_step = n_iter // 20 if n_iter > 20 else 1
        for i in range(n_iter):
            with tf.GradientTape() as tape:
                dist = self(self.t_data, training = True)
                loss = self.compute_loss(dist, self.y_data)
            grads = tape.gradient(loss, self.trainable_variables)
            #print(grads)
            if (i % print_step == 0):
                #mu, sigma = self.transform_vars(self.mu_0, self.sigma_0)
                print(f'Iteration {i}: loss {loss}, {self.params_to_str()}')
    #                 print(grads)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss

    def predict(self, x):
#         sampling_fraction = (tf.math.tanh(self.sampling_fraction) + 1) / 2
        return self(x).mean().numpy()

def fit_peak(adata, gene):
    x = adata.obs['pseudo_t']
    y = adata[:, gene].X.flatten().astype('float32')
    x_y_mask = (~np.isnan(x)) & (~np.isnan(y))
    x = x[x_y_mask]
    y = y[x_y_mask]
    plt.scatter(x, y)
    gene_model = GeneSwitchModel(x, y, name = 'gene_switch_model')
    print(gene_model.trainable_variables)
    gene_model.train(n_iter = 1000)

    x1 = np.linspace(0, 1, 100)
    total_count = gene_model.predict(x1)
    sampling_fraction = gene_model.sampling_fraction().numpy()
    y1 = total_count * sampling_fraction
    # print(f'Sampling fraction: {sampling_fraction}')
    plt.plot(x1, y1, 'r')


def fit_peaks(adata, gene_list, layer = None, plots = False):
    x = np.asarray(adata.obs['pseudo_t'])
    if layer is None:
        y = adata[:, gene_list].X.astype('float32')
    else:
        y = adata[:, gene_list].layers[layer].astype('float32')
    mask = ~np.isnan(x)
    x = x[mask]
    y = y[mask, :]

    gene_model = GeneSwitchModel(x, y, name = 'gene_switch_model')
    gene_model.train(n_iter = 300, learning_rate = 0.05)

    result = pd.DataFrame({
        'ampl': gene_model.ampl.numpy(),
        'mu': gene_model.mu.numpy(),
        'sigma': gene_model.sigma.numpy()
    }, index = gene_list)

    if plots:
        x1 = np.linspace(0, 1, 100).reshape((-1, 1))
        y1 = gene_model.predict(x1)
        fg = figure_grid(n_col= 4, n_total= len(gene_list))
        gene_index = 0
        for gene, ax in zip(gene_list, fg):
            ax.scatter(x, y[:, gene_index])
            ax.plot(x1, y1[:, gene_index], 'r')
            ax.set_title(gene)
            gene_index += 1

    return result
