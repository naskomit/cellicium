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


class GaussianDistributionLayerLearnablePrior(tfk.layers.Layer):
    def __init__(self, n_dim : int, n_cat : int, name : str = None):
        super().__init__(name = name)
        shape = (n_dim, n_cat)

        self.prior_mean = tf.Variable(
            initial_value = tf.random_normal_initializer()(shape = shape),
            name = f'{name}_prior_mean'
        )
        # Log variance for the background rate
        self.prior_log_var = tf.Variable(
            initial_value = tf.clip_by_value(tf.random_normal_initializer()(shape = shape), -10, 1),
            name = f'{name}_prior_logvar'
        )

    def call(self, inputs, training = False):
        mean, logvar, cat = inputs
        post_sd = tf.exp(logvar / 2)
        posterior = tfp.distributions.Normal(loc = mean, scale = post_sd)

        prior_mean = tf.matmul(cat, tf.transpose(self.prior_mean))
        prior_log_var = tf.matmul(cat, tf.transpose(self.prior_log_var))
        prior = tfp.distributions.Normal(loc = prior_mean, scale = tf.exp(prior_log_var / 2))

        kl_divergence = tfp.distributions.kl_divergence(distribution_a = posterior, distribution_b = prior)
        kl_divergence = tf.reduce_mean(kl_divergence)
        if training:
            eps = tfp.distributions.Normal(loc = 0, scale = tf.ones_like(post_sd)).sample()
            z = mean + eps * post_sd
        else:
            tf.print('Predicting using the mean')
            z = mean

        return {'z': z, 'kl_divergence': kl_divergence}


class NormalDistributionLayer(tfk.layers.Layer):
    def __init__(self, n_dim : int, name : str = None):
        super().__init__(name = name)
        self.n_dim = n_dim

    def call(self, input, training = False):
        mean, logvar = input
        post_sd = tf.exp(logvar / 2)
        posterior = tfp.distributions.Normal(loc = mean, scale = post_sd)

        prior_mean = tf.zeros_like(posterior.loc)
        prior_std = tf.ones_like(posterior.scale)
        prior = tfp.distributions.Normal(loc = prior_mean, scale = prior_std)

        kl_divergence = tfp.distributions.kl_divergence(distribution_a = posterior, distribution_b = prior)
        kl_divergence = tf.reduce_mean(kl_divergence)

        if training:
            eps = tfp.distributions.Normal(loc = 0, scale = tf.ones_like(post_sd)).sample()
            z = mean + eps * post_sd
        else:
            tf.print('Predicting using the mean')
            z = mean

        return {'z': z, 'kl_divergence': kl_divergence}


class ADT_Autoencoder(tfk.Model):
    def __init__(self, dim_x : int, dim_z : int, n_batches : int, n_layers : int):
        super().__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.n_batches = n_batches
        self.n_hidden_dim : int = 128
        self.n_layers = n_layers
        self.dropout = 0.2

        self.x_dispersion = tf.Variable(
            initial_value = tf.random_uniform_initializer(minval = 0., maxval = 2.)(shape = [self.dim_x, self.n_batches]),
            name = 'protein dispersion'
        )

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.loss_trackers = {
            'total_loss': {'metric': tfk.metrics.Mean(name = 'total_loss'), 'gain': 1},
            'reconstr_loss': {'metric': tfk.metrics.Mean(name = 'reconstr_loss'), 'gain': 1},
            'kl_weight': {'metric': tfk.metrics.Mean(name = 'kl_weight'), 'gain': 1},
            'z_kl': {'metric': tfk.metrics.Mean(name = 'z_kl'), 'gain': 1},
            'rate_back_kl': {'metric': tfk.metrics.Mean(name = 'z_kl'), 'gain': 1}

        }

        self.training_plan : nnu.TrainingPlan = None

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
        self.z_dist = NormalDistributionLayer(n_dim = self.dim_z, name = 'z_dist')
        z_out = self.z_dist((z_dist_mean, z_dist_logvar))
        z_sample = tfk.layers.Softmax(axis = -1, name = 'z_sample')(z_out['z'])
        encoder = tfk.Model(inputs = [x_input, batch_input],
                            outputs = {'z_dist_mean': z_dist_mean, 'z_dist_logvar': z_dist_logvar,
                                       'z': z_sample, 'z_kl_div': z_out['kl_divergence']})
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

        self.rate_back_dist = GaussianDistributionLayerLearnablePrior(
            n_dim = self.dim_x, n_cat = self.n_batches, name = 'rate_back_dist')
        rate_back_out = self.rate_back_dist((
            back_dist_mean, back_dist_logvar, batch_input
        ))

        def f1(y):
            y = tf.clip_by_value(y, clip_value_min = -100.0, clip_value_max = 20.0)
            result = tf.math.exp(y)
            return result

        rate_back = tfk.layers.Lambda(f1, name = 'rate_back')(rate_back_out['z'])
        # rate_back = tfk.layers.Lambda(lambda y : tf.math.exp(y), name = 'rate_back')(rate_back)
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

        outputs = {'rate_back': rate_back, 'rate_fore': rate_fore, 'mixing': mixing,
                   'fore_scale': fore_scale, 'rate_back_kl_div': rate_back_out['kl_divergence']}

        decoder = tfk.Model(inputs = [z_input, batch_input], outputs = outputs)
        return decoder

    def _inference(self, x, batch_ohe):
        x = tf.math.log(1 + x)
        z_data = self.encoder((x, batch_ohe))
        return z_data

    def _generative(self, inference_outputs, batch_ohe):
        z = inference_outputs['z']
        px = self.decoder((z, batch_ohe)) #, log_pro_back_mean
        disp_back = tf.linalg.matmul(batch_ohe, tf.transpose(self.x_dispersion))
        px['disp_back'] = tf.math.exp(disp_back)
        # px['disp_back'] = tf.broadcast_to(tf.math.sigmoid(self.x_dispersion), tf.shape(px['rate_back']))
        # px['disp_fore'] = tf.broadcast_to(tf.math.sigmoid(self.x_dispersion), tf.shape(px['rate_fore']))
        return px

    def _reconstruction_loss(self, x_in, px):
        px_cond = NegativeBinomialMixture(mu1 = px['rate_back'], mu2 = px['rate_fore'],
                                          theta1 = px['disp_back'], mixture_logits = px['mixing'])
        reconstr_loss = -px_cond.log_prob(x_in)
        reconstr_loss = tf.reduce_mean(reconstr_loss)
        return reconstr_loss

    def _compute_loss(self, inputs, inference_outputs, generative_outputs):
        x, batch = inputs
        reconstr_loss = self._reconstruction_loss(x, generative_outputs)

        reglr_losses = {'z_kl': inference_outputs['z_kl_div'],
                        'rate_back_kl': generative_outputs['rate_back_kl_div'],
                        'other_losses': tf.reduce_sum(self.losses)}
        current_kl_weight = self.training_plan.current_kl_weight()
        total_loss = reconstr_loss + current_kl_weight  * (
                reglr_losses['z_kl'] + reglr_losses['rate_back_kl'])
        losses  = {'total_loss': total_loss, 'reconstr_loss': reconstr_loss, **reglr_losses,
                   'kl_weight': 1e3 * current_kl_weight}
        return losses

    def call(self, inputs):
        x, batch = inputs
        batch_ohe = tf.one_hot(batch, depth = self.n_batches, dtype = tf.float32)
        inference_outputs = self._inference(x, batch_ohe)
        generative_outputs = self._generative(inference_outputs, batch_ohe)
        losses = self._compute_loss(inputs, inference_outputs, generative_outputs)
        return inference_outputs, generative_outputs, losses

    def set_training_plan(self, plan : nnu.TrainingPlan):
        self.training_plan = plan

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        # self.epochs_trained.assign_add(1)
        # print(f'Epoch: {self.epochs_trained}')
        with tf.GradientTape() as tape:
            inference_outputs, generative_outputs, losses = self(data, training = True)
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
        _, _, losses = self(data, training = True)
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
        inference_outputs, generative_outputs, losses = self(data)
        mu = inference_outputs['z_dist_mean']
        mu = tf.math.softmax(mu, axis = -1)
        return mu.numpy()

    def predict_foreground(self, data, scale = False):
        _, px, _ = self(data)
        protein_mixing = 1 / (1 + tf.math.exp(-px["mixing"]))
        protein_val = px["rate_fore"] * (1 - protein_mixing)
        if scale:
            protein_val, _ = tf.linalg.normalize(protein_val, ord = 1, axis = -1)
        return protein_val.numpy()

    def predict_background(self, data):
        _, px, _ = self(data)
        protein_mixing = 1 / (1 + tf.math.exp(-px["mixing"]))
        protein_val = px["rate_back"] * protein_mixing
        return protein_val.numpy()

    def predict_mixing(self, data):
        _, px, _ = self(data)
        protein_mixing = 1 / (1 + tf.math.exp(-px["mixing"]))
        return protein_mixing.numpy()

    def predict_values(self, data):
        inference_outputs, generative_outputs, losses = self(data)
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

class VAETrainingPlan(nnu.TrainingPlan):
    pass


class AutoencoderManager(base.ModelManagerBase):
    def __init__(self, dim_x : int, dim_z : int, n_batches : int, n_layers : int):
        super().__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.n_batches = n_batches
        self.n_layers = n_layers

    def build_model(self, **kwargs) -> tfk.Model:
        model = ADT_Autoencoder(
            dim_x = self.dim_x, dim_z = self.dim_z, n_batches = self.n_batches, n_layers = self.n_layers
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

    def train(self, adata : sc.AnnData, training_plan : VAETrainingPlan, **kwargs):
        layer = kwargs.pop('layer', None)
        X, batch = self.create_inputs(adata, layer)
        dataset = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(X),
            tf.data.Dataset.from_tensor_slices(batch)
        ))
        self.do_train(dataset, adata.n_obs, training_plan, **kwargs)

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
