import scipy.sparse as sparse
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp
import scanpy as sc
import os
from . import base
from . import nn_utils as nnu

debug_data = {}

def negative_binomial_2comp_mix(mu1 : tf.Tensor, mu2 : tf.Tensor, theta1 : tf.Tensor, theta2  : tf.Tensor,
                                mixture_logits: tf.Tensor = None):
    mixture_logits = tf.stack([mixture_logits, - mixture_logits], axis = -1)

    return tfp.distributions.Mixture(
        cat = tfp.distributions.Categorical(logits = mixture_logits),
        components = [
            tfp.distributions.NegativeBinomial.experimental_from_mean_dispersion(mean = mu1, dispersion = theta1),
            tfp.distributions.NegativeBinomial.experimental_from_mean_dispersion(mean = mu2, dispersion = theta2)
        ]
    )

def normal_distribution_block(x, n_dim: int, kl_weight : float, prior  = None, name : str = None):
    if prior is None:
        prior = tfp.distributions.Independent(
            tfp.distributions.Normal(loc = tf.zeros(n_dim), scale = 1.),
            reinterpreted_batch_ndims = 1)

    x = tfp.layers.IndependentNormal(
        n_dim,
        convert_to_tensor_fn = tfp.distributions.Distribution.sample,
        activity_regularizer = tfp.layers.KLDivergenceRegularizer(prior, weight = kl_weight),
        name= name
    )(x)

    return x

class NegativeBinomialMixture():
    def __init__(self, mu1: tf.Tensor, mu2: tf.Tensor, theta1: tf.Tensor, mixture_logits: tf.Tensor):
        self.mu1 = mu1
        self.theta1 = theta1
        self.mu2 = mu2
        self.mixture_logits = mixture_logits

    def log_prob(self, x : tf.Tensor) -> tf.Tensor:
        eps = 1e-8
        theta = self.theta1
        mu_1 = self.mu1
        mu_2 = self.mu2
        pi_logits = self.mixture_logits
        log_theta_mu_1_eps = tf.math.log(theta + mu_1 + eps)
        log_theta_mu_2_eps = tf.math.log(theta + mu_2 + eps)
        lgamma_x_theta = tf.math.lgamma(x + theta)
        lgamma_theta = tf.math.lgamma(theta)
        lgamma_x_plus_1 = tf.math.lgamma(x + 1)

        log_nb_1 = (
                theta * (tf.math.log(theta + eps) - log_theta_mu_1_eps)
                + x * (tf.math.log(mu_1 + eps) - log_theta_mu_1_eps)
                + lgamma_x_theta
                - lgamma_theta
                - lgamma_x_plus_1
        )

        log_nb_2 = (
                theta * (tf.math.log(theta + eps) - log_theta_mu_2_eps)
                + x * (tf.math.log(mu_2 + eps) - log_theta_mu_2_eps)
                + lgamma_x_theta
                - lgamma_theta
                - lgamma_x_plus_1
        )

        logsumexp = tf.math.reduce_logsumexp(tf.stack((log_nb_1, log_nb_2 - pi_logits)), axis = 0)
        softplus_pi = tf.math.softplus(-pi_logits)

        log_mixture_nb = logsumexp - softplus_pi

        return log_mixture_nb


class NormalDistributionLayer(tfk.layers.Layer):
    SAMPLE = 0
    MEAN = 1

    PRIOR_NORMAL = 0
    PRIOR_LEARN = 1

    def __init__(self, n_dim : int, n_cat : int = None, prior : int = PRIOR_NORMAL, name : str = None):
        super().__init__()
        # self.mode = self.SAMPLE
        self.n_dim = n_dim
        if n_cat is None:
            shape = (n_dim,)
        else:
            shape = (n_dim, n_cat)
        self.prior = prior
        self._kl_divergence = None

        if prior == self.PRIOR_LEARN:
            # Mean for the background rate
            self.prior_mean = tf.Variable(
                initial_value = tf.random_normal_initializer()(shape = shape),
                name = name + '_prior_mean'
            )
            # Log variance for the background rate
            self.prior_log_var = tf.Variable(
                initial_value = tf.clip_by_value(tf.random_normal_initializer()(shape = shape), -10, 1),
                name = name + '_prior_log_var'
            )
        elif prior == self.PRIOR_NORMAL:
            pass
        else:
            raise ValueError('Prior must be one of PRIOR_NORMAL or PRIOR_LEARN')

    @property
    def kl_divergence(self):
        return self._kl_divergence

    def call(self, input, cat = None):
        mean, logvar = input
        dist = tfp.distributions.Normal(loc = mean, scale = tf.exp(logvar / 2))
        self._kl_divergence = self.compute_kl_divergence(cat, dist)
        return dist.sample()
        # else:
        #     self._kl_divergence = 0
        #     return mean

    @tf.function
    def compute_kl_divergence(self, cat, posterior):
        if self.prior == self.PRIOR_LEARN:
            prior_mean = tf.matmul(cat, tf.transpose(self.prior_mean))
            prior_log_var = tf.matmul(cat, tf.transpose(self.prior_log_var))
            prior = tfp.distributions.Normal(loc = prior_mean, scale = tf.exp(prior_log_var / 2))
        else:
            prior_mean = tf.zeros_like(posterior.loc)
            prior_std = tf.ones_like(posterior.scale)
            prior = tfp.distributions.Normal(loc = prior_mean, scale = prior_std)

        result = tfp.distributions.kl_divergence(distribution_a = posterior, distribution_b = prior)
        result = tf.reduce_mean(tf.reduce_sum(result, axis = -1), axis = 0)

        # try:
        # except tf.errors.InvalidArgumentError as e:
        #     debug_data['prior'] = prior
        #     debug_data['posterior'] = posterior
        #     raise e
        # # tf.print('============ KL Divergence ============')
        # # tf.print(result)

        return result


def softplus(x):
    return tfk.layers.Lambda(lambda y : tf.math.softplus(y))(x)


class ADT_Autoencoder(tfk.Model):
    def __init__(self, dim_x : int, dim_z : int, n_batches : int, kl_weight : float, n_layers : int, warmup :  int):
        super().__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.n_batches = n_batches
        self.n_hidden_dim : int = 128
        self.n_layers = n_layers
        self.kl_weight = kl_weight
        self.warmup = warmup
        self.dropout = 0.3

        self.x_dispersion = tf.Variable(
            initial_value = tf.random_uniform_initializer(minval = 0., maxval = 2.)(shape = [self.dim_x, self.n_batches]),
            name = 'protein dispersion'
        )

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.loss_trackers = {
            'total_loss': {'metric': tfk.metrics.Mean(name = 'total_loss'), 'gain': 1},
            'reconstr_loss': {'metric': tfk.metrics.Mean(name = 'reconstr_loss'), 'gain': 1},
            'reglr_loss': {'metric': tfk.metrics.Mean(name = 'reglr_loss'), 'gain': 1}
        }

        self.epochs_trained = tf.Variable(0, trainable = False, dtype = tf.int32)

    def block1(self, x, n_dims, dropout, n_layers = 1, name = None):
        def make_name(i, postfix):
            if name is None:
                return None
            else:
                return f'{name}_{i}_{postfix}'
        for i in range(n_layers):
            x = tfk.layers.Dense(units = n_dims, name = make_name(i, 'dense'))(x)
            x = tfk.layers.BatchNormalization(name = make_name(i, 'batch_norm'))(x)
            x = tfk.layers.ReLU(name = make_name(i, 'relu'))(x)
            x = tfk.layers.Dropout(rate = dropout, name = make_name(i, 'dropout'))(x)
        return x

    def build_encoder(self) -> tfk.Model:
        x_input = tfk.Input(shape = self.dim_x, name = 'x_input')
        batch_input = tfk.Input(shape = self.n_batches, name = 'batch input')
        encoder_input = tfk.layers.Concatenate(axis = -1, name = 'encoder_input')([x_input, batch_input])
        x1 = self.block1(encoder_input, n_dims = self.n_hidden_dim, dropout = self.dropout,
                         n_layers= self.n_layers, name = 'x1_1')
        x1 = self.block1(x1, n_dims = self.n_hidden_dim, dropout = self.dropout,
                         n_layers= self.n_layers, name = 'x1_2')
        z_dist_mean = tfk.layers.Dense(units = self.dim_z, name = 'z_dist_mean')(x1)
        z_dist_logvar = tfk.layers.Dense(units = self.dim_z, name = 'z_dist_logvar')(x1)
        self.z_dist = NormalDistributionLayer(n_dim = self.dim_z,
                                              prior = NormalDistributionLayer.PRIOR_NORMAL,
                                              name = 'z_dist')
        z = self.z_dist((z_dist_mean, z_dist_logvar))
        z_sample = tfk.layers.Softmax(axis = -1, name = 'z_sample')(z)
        encoder = tfk.Model(inputs = [x_input, batch_input],
                            outputs = {'z_dist_mean': z_dist_mean, 'z_dist_logvar': z_dist_logvar, 'z': z_sample})
        return encoder

    def build_decoder(self) -> tfk.Model:
        z_input = tfk.Input(shape = self.dim_z, name = 'z_input')
        batch_input = tfk.Input(shape = self.n_batches, name = 'batch input')

        decoder_input = tfk.layers.Concatenate(axis = -1, name = 'decoder_input')([z_input, batch_input])

        z_back = self.block1(decoder_input, n_dims = self.n_hidden_dim, dropout = self.dropout,
                             n_layers= self.n_layers, name = 'z_back')
        z_back = tfk.layers.Concatenate(axis = -1, name = 'z_back_concat')([z_back, decoder_input])
        z_fore = self.block1(decoder_input, n_dims = self.n_hidden_dim, dropout = self.dropout,
                             n_layers= self.n_layers, name = 'z_fore')
        z_fore = tfk.layers.Concatenate(axis = -1, name = 'z_fore_concat')([z_fore, decoder_input])

        back_dist_mean = self.block1(z_back, n_dims = self.dim_x, dropout = self.dropout,
                                       n_layers= self.n_layers, name = 'back_dist_mean')
        back_dist_logvar = self.block1(z_back, n_dims = self.dim_x, dropout = self.dropout,
                                     n_layers= self.n_layers, name = 'back_dist_logvar')

        # rate_back = normal_distribution_block(back_dist_params, n_dim = self.dim_x, kl_weight = self.kl_weight,
        #                                       name = 'rate_back_dist')
        # n_dim : int, n_cat : int = None, prior : int = PRIOR_NORMAL
        self.rate_back_dist = NormalDistributionLayer(n_dim = self.dim_x, n_cat = self.n_batches,
                                                      prior = NormalDistributionLayer.PRIOR_LEARN,
                                                      name = 'rate_back_dist')
        rate_back = self.rate_back_dist((
            back_dist_mean, back_dist_logvar
        ), cat = batch_input)
        rate_back = tfk.layers.Lambda(lambda y : tf.math.exp(y), name = 'rate_back')(rate_back)
        fore_scale = self.block1(z_fore, n_dims = self.dim_x, dropout = self.dropout,
                                 n_layers= self.n_layers, name = 'fore_scale')

        def rate_fore_fn(inputs):
            _fore_scale, _rate_back = inputs
            return (_fore_scale + 1 + 1e-8) * _rate_back

        rate_fore = tfk.layers.Lambda(rate_fore_fn)((fore_scale, rate_back))

        p_mixing = self.block1(decoder_input, n_dims = self.n_hidden_dim, dropout = self.dropout,
                               n_layers= self.n_layers, name = 'p_mixing')
        p_mixing = tfk.layers.Concatenate(axis = -1, name = 'p_mixing_concat')([p_mixing, decoder_input])
        mixing = self.block1(p_mixing, n_dims = self.dim_x, dropout = self.dropout,
                             n_layers= self.n_layers, name = 'mixing')

        outputs = {'rate_back': rate_back, 'rate_fore': rate_fore, 'mixing': mixing, 'fore_scale': fore_scale}

        decoder = tfk.Model(inputs = [z_input, batch_input], outputs = outputs)
        return decoder

    def _inference(self, x, batch_ohe):
        x = tf.math.log(1 + x)
        # px_back_alpha_prior = tf.linalg.matmul(batch_ohe, self.background_pro_alpha)
        # px_back_beta_prior = tf.linalg.matmul(batch_ohe, tf.math.exp(self.background_pro_log_beta))
        # self.back_mean_prior =
        z_data = self.encoder((x, batch_ohe))
        # if sample:
        #     self.z_dist.set_mode(NormalDistributionLayer.SAMPLE)
        #     #tf.print('Sample')
        # else:
        #     self.z_dist.set_mode(NormalDistributionLayer.MEAN)
        #     #tf.print('Mean')
        return z_data

    def _generative(self, inference_outputs, batch_ohe):
        # if sample:
        #     self.rate_back_dist.set_mode(NormalDistributionLayer.SAMPLE)
        # else:
        #     self.rate_back_dist.set_mode(NormalDistributionLayer.MEAN)
        z = inference_outputs['z']
        px = self.decoder((z, batch_ohe)) #, log_pro_back_mean
        disp_back = tf.linalg.matmul(batch_ohe, tf.transpose(self.x_dispersion))
        px['disp_back'] = tf.math.exp(disp_back)
        # px['disp_back'] = tf.broadcast_to(tf.math.sigmoid(self.x_dispersion), tf.shape(px['rate_back']))
        # px['disp_fore'] = tf.broadcast_to(tf.math.sigmoid(self.x_dispersion), tf.shape(px['rate_fore']))
        return px

    def _reconstruction_loss(self, x_in, px):
        # px_cond = negative_binomial_2comp_mix(
        #     mu1 = px['rate_back'], mu2 = px['rate_fore'], theta1 = px['disp_back'], theta2 = px['disp_fore'],
        #     mixture_logits = px['mixing']
        # )
        # Could be used for debugging
        # try:
        # except tf.errors.InvalidArgumentError as e:
        #     debug_data['px'] = px
        #     debug_data['x_in'] = x_in
        #     raise e
        px_cond = NegativeBinomialMixture(mu1 = px['rate_back'], mu2 = px['rate_fore'],
                                          theta1 = px['disp_back'], mixture_logits = px['mixing'])
        reconstr_loss = -px_cond.log_prob(x_in)
        reconstr_loss = tf.reduce_mean(reconstr_loss)
        return reconstr_loss

    def _regularization_loss(self):
        losses = tf.reduce_sum(self.losses)
        kl_divergence = self.z_dist.kl_divergence + self.rate_back_dist.kl_divergence
        if self.warmup > 0:
            klweight = self.kl_weight
        return losses + self.kl_weight * kl_divergence


    def _compute_loss(self, inputs, inference_outputs, generative_outputs):
        x, batch = inputs
        reconstr_loss = self._reconstruction_loss(x, generative_outputs)
        # TODO add other losses: regularization, kl-divergence etc.
        reglr_loss = self._regularization_loss()
        total_loss = reconstr_loss + reglr_loss
        losses  = {'reconstr_loss': reconstr_loss, 'reglr_loss': reglr_loss, 'total_loss': total_loss}
        return losses

    def call(self, inputs, sample = True):
        x, batch = inputs
        batch_ohe = tf.one_hot(batch, depth = self.n_batches, dtype = tf.float32)
        inference_outputs = self._inference(x, batch_ohe)
        generative_outputs = self._generative(inference_outputs, batch_ohe)
        losses = self._compute_loss(inputs, inference_outputs, generative_outputs)
        return inference_outputs, generative_outputs, losses

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        # self.epochs_trained.assign_add(1)
        # print(f'Epoch: {self.epochs_trained}')
        with tf.GradientTape() as tape:
            inference_outputs, generative_outputs, losses = self(data)
        total_loss = losses['total_loss']
        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(total_loss, self.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_weights)
        )

        # # Let's update and return the training loss metric.
        for k, v in self.loss_trackers.items():
            v['metric'].update_state(losses[k])
        return {k: v['metric'].result() for k, v in self.loss_trackers.items()}

    def test_step(self, data):
        _, _, losses = self(data)
        # Let's update and return the loss metric.
        for k, v in self.loss_trackers.items():
            v['metric'].update_state(losses[k])
        return {k: v['metric'].result() for k, v in self.loss_trackers.items()}

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [v['metric'] for k, v in self.loss_trackers.items()]

    def predict_latent(self, data):
        inference_outputs, generative_outputs, losses = self(data, sample = False)
        mu = inference_outputs['z_dist_mean']
        mu = tf.math.softmax(mu, axis = -1)
        return mu.numpy()

    def predict_foreground(self, data, scale = False):
        _, px, _ = self(data, sample = False)
        protein_mixing = 1 / (1 + tf.math.exp(-px["mixing"]))
        protein_val = px["rate_fore"] * (1 - protein_mixing)
        if scale:
            protein_val, _ = tf.linalg.normalize(protein_val, ord = 1, axis = -1)
        return protein_val.numpy()

    def predict_background(self, data):
        _, px, _ = self(data, sample = False)
        protein_mixing = 1 / (1 + tf.math.exp(-px["mixing"]))
        protein_val = px["rate_back"] * protein_mixing
        return protein_val.numpy()

    def predict_mixing(self, data):
        _, px, _ = self(data, sample = False)
        protein_mixing = 1 / (1 + tf.math.exp(-px["mixing"]))
        return protein_mixing.numpy()

    def predict_values(self, data):
        inference_outputs, generative_outputs, losses = self(data, sample = False)
        mu = inference_outputs['z_dist_mean']
        # mu = tf.math.softmax(mu, axis = -1)
        mixing = 1 / (1 + tf.math.exp(-generative_outputs["mixing"]))
        foreground = generative_outputs["rate_fore"] * (1 - mixing)
        background = generative_outputs["rate_back"] * mixing
        scaled, _ = tf.linalg.normalize(foreground, ord = 1, axis = -1)

        result = sc.AnnData(X = scaled.numpy())
        result.layers['rate_back'] = generative_outputs['rate_back'].numpy()
        result.layers['rate_fore'] = generative_outputs['rate_fore'].numpy()
        result.layers['fore_scale'] = generative_outputs['fore_scale'].numpy()
        result.layers['mixing'] = mixing.numpy()

        result.layers['foreground'] = foreground.numpy()
        result.layers['background'] = background.numpy()

        result.obsm['z'] = mu.numpy()
        result.varm['mean_back'] = tf.exp(self.rate_back_dist.prior_mean).numpy()

        return result


class AutoencoderManager(base.ModelManagerBase):
    def __init__(self, dim_x : int, dim_z : int, n_batches : int, kl_weight : float, n_layers : int, warmup : int):
        super().__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.warmup = warmup
        self.n_batches = n_batches
        self.kl_weight = kl_weight
        self.n_layers = n_layers

    def build_model(self, **kwargs) -> tfk.Model:
        model = ADT_Autoencoder(
            dim_x = self.dim_x, dim_z = self.dim_z, n_batches = self.n_batches,
            kl_weight = self.kl_weight, n_layers = self.n_layers, warmup = self.warmup
        )
        return model

    def save_weights(self, dir : str, name : str = 'ADT_VAE_weights.h5'):
        self.model.save_weights(os.path.join(dir, name), save_format = 'hdf5')

    def load_weights(self, adata, dir : str, name : str = 'ADT_VAE_weights.h5'):
        self.model = self.build_model()
        # TODO
        self.latent(adata)
        self.model.load_weights(os.path.join(dir, name))

    @staticmethod
    def create_inputs(adata : sc.AnnData, layer = 'counts'):
        def ensure_dense(X):
            X_dense = X.todense() if sparse.issparse(X) else X
            X_dense = np.asarray(X_dense)
            return X_dense

        X = adata.X if layer is None else adata.layers[layer]
        X = ensure_dense(X)
        batch = pd.Categorical(adata.obs['batch'])
        batch = batch.codes.astype('uint8')
        return X, batch

    def train(self, adata : sc.AnnData, epochs, lr, **kwargs):
        X, batch = self.create_inputs(adata)
        dataset = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(X),
            tf.data.Dataset.from_tensor_slices(batch)
        ))
        kwargs = {**kwargs, 'epochs': epochs, 'lr': lr, 'validation_split': 0}
        self.do_train(dataset, adata.n_obs, **kwargs)

    def latent(self, adata : sc.AnnData):
        X, batch  = self.create_inputs(adata)
        mu = self.model.predict_latent((X, batch))
        result = sc.AnnData(X = mu, obs = adata.obs)
        return result

    def foreground(self, adata : sc.AnnData, scale = False):
        X, batch  = self.create_inputs(adata)
        fore = self.model.predict_foreground((X, batch), scale)
        result = sc.AnnData(X = fore, obs = adata.obs, var = adata.var)
        return result

    def background(self, adata : sc.AnnData):
        X, batch  = self.create_inputs(adata)
        back = self.model.predict_background((X, batch))
        result = sc.AnnData(X = back, obs = adata.obs, var = adata.var)
        return result

    def mixing(self, adata : sc.AnnData):
        X, batch  = self.create_inputs(adata)
        mix = self.model.predict_mixing((X, batch))
        result = sc.AnnData(X = mix, obs = adata.obs, var = adata.var)
        return result

    def predict_values(self, adata : sc.AnnData):
        X, batch  = self.create_inputs(adata)
        result = self.model.predict_values((X, batch))
        result.obs = adata.obs.copy()
        result.var = adata.var.copy()
        return result
