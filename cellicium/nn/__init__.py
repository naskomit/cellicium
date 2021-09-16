import datetime, os, time
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from .peak_finding import fit_peak

# class NormalDistributionVariationalLayer(tfk.layers.Layer):
#     def __init__(self, reconstr_weight = 1.0, kl_weight = 1.0):
#         super().__init__()
#         self.reconstr_weight = reconstr_weight
#         self.kl_weight = kl_weight
#
#     def build(self, inputs):
#         print("Inputs")
#         print(inputs)
#         x_in = tfk.Input(shape = inputs[0].as_list()[-1])
#         x_out = tfk.Input(shape = inputs[1].as_list()[-1])
#         z_mean = tfk.Input(shape = inputs[2].as_list()[-1])
#         z_log_var = tfk.Input(shape = inputs[3].as_list()[-1])
#         reconstr_loss = self.reconstr_weight * tf.losses.mean_squared_error(x_in, x_out)
#         kl_loss = - 0.5 * self.kl_weight * tf.math.reduce_sum(
#             1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
#             axis = -1
#         )
#         loss = tf.reduce_mean(reconstr_loss + kl_loss)
#         self.add_loss(loss, inputs = inputs)
#         self.add_metric(tfk.metrics.MeanSquaredError()(x_in, x_out), name = 'reconstr_loss_1')
#         self.model = tfk.Model([x_in, x_out, z_mean, z_log_var], loss)
#         return self.model
#
#     # def call(self, inputs):
#     #     x_in = inputs[0]
#     #     x_out = inputs[1]
#     #     loss = self.model(inputs)
#         # z_mean = inputs[2]
#         # z_log_var = inputs[3]
#         # reconstr_loss = self.reconstr_weight * tf.losses.mean_squared_error(x_in, x_out)
#         # kl_loss = - 0.5 * self.kl_weight * tf.math.reduce_sum(
#         #     1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
#         #     axis = -1
#         # )
#         # loss = tf.reduce_mean(reconstr_loss + kl_loss)
#         # self.add_loss(loss, inputs = inputs)
#         # self.add_metric(tfk.metrics.MeanSquaredError()(x_in, x_out), name = 'reconstr_loss_1')
# #        return x_out
#
#
# class VariationalAutoencoder1(object):
#     def __init__(self, n_var, latent_dim = 2, batch_size = 16):
#         super().__init__()
#         self.n_var = n_var
#         self.latent_dim = latent_dim
#         self.batch_size = batch_size
#         self.build()
#
#
#     def build(self):
#         # Input
#         self.encoder_input = tfk.Input(shape = (self.n_var,), name = 'encoder_input')
#         # Encoder
#         x = tfk.layers.Dense(32, activation = 'relu', name = 'encoder_1')(self.encoder_input)
#         z_mean = tfk.layers.Dense(self.latent_dim, name = 'encoder_mean')(x)
#         z_log_var = tfk.layers.Dense(self.latent_dim, name = 'encoder_variance')(x)
#         self.encoder = tfk.Model(self.encoder_input, [z_mean, z_log_var])
#         # Sampler
#         def sampling(args):
#             z_mean, z_log_var = args
#             epsilon = tf.random.normal(
#                 shape = (tf.shape(z_mean)[0], self.latent_dim),
#                 mean = 0.0, stddev = 1.0)
#             return z_mean + tf.exp(z_log_var) * epsilon
#         z = tfk.layers.Lambda(sampling)([z_mean, z_log_var])
#         # Decoder
#         self.decoder_input = tfk.Input(z.shape[1])
#         self.decoder_layer = tfk.layers.Dense(self.n_var, activation = 'relu', name = 'decoder_1')
#         #self.time_step_layer = tfk.layers.Dense(self.latent_dim)
#         x_reconstr = self.decoder_layer(self.decoder_input)
#         #x_next = self.decoder_layer(self.time_step_layer(self.decoder_input))
#         self.decoder = tfk.Model(self.decoder_input, x_reconstr)
#         reconstr_output = self.decoder(z)
#
#         self.loss_output = NormalDistributionVariationalLayer()(
#             [self.encoder_input, reconstr_output, z_mean, z_log_var]
#         )
#
#         self.model = tfk.Model(self.encoder_input, [self.loss_output])
#
#     def train(self, x_in):
#
#         self.model.compile(
#             optimizer = 'rmsprop',
#             loss = None
# #             metrics = ['cosine_similarity', 'mean_squared_error', 'kullback_leibler_divergence']
#         )
#         self.model.summary()
#         fit_result = self.model.fit(
#             x = x_in, y = None,
#             shuffle = True, epochs = 100,
#             batch_size = self.batch_size,
#             validation_split = 0.3, verbose = True)
#         return fit_result
#
#
#
# ######################################################################################
# class DirectModel():
#     def __init__(self, dim_z):
#         self.dim_z = dim_z
#
#     def create_params_layer(self, name = 'z_params'):
#         return tfk.layers.Dense(self.dim_z, activation = None, name = name)
#
#     def create_sampler_layer(self):
#         return (lambda x: (x, x))
#
#     def kl_divergence(self, data):
#         return 0
#
# class GaussianModel():
#     class Sampler(tfk.layers.Layer):
#         def call(self, inputs, training = None, mask = None):
#             mu, rho = tf.split(inputs, num_or_size_splits = 2, axis = 1)
#             sd = tf.math.log(1 + tf.math.exp(rho))
#             batch_size = tf.shape(mu)[0]
#             dim_z = tf.shape(mu)[1]
#             if training:
#                 z_sample = mu + sd * tf.random.normal(shape = (batch_size, dim_z))
#             else:
#                 z_sample = mu
#             z_params = {'mu': mu, 'sd': sd}
#             return z_sample, z_params
#
#     def __init__(self, dim_z):
#         self.dim_z = dim_z
#
#     def create_params_layer(self, name = 'z_params'):
#         return tfk.layers.Dense(2 * self.dim_z, activation = None, name = name)
#
#     def create_sampler_layer(self):
#         return self.Sampler()
#
#     def kl_divergence(self, data):
#         z_sample, z_params = data
#         mu = z_params['mu']; sd = z_params['sd']
#         result = - 0.5 * tf.math.reduce_sum(1 + tf.math.log(tf.math.square(sd)) -
#             tf.math.square(mu) - tf.math.square(sd), axis = 1)
#         #print("kl_divergence(raw) shape: ", kl_divergence.shape)
#         return tf.math.reduce_mean(result)
# ###########
# # class GaussianCircularModel():
# #     class Sampler(tfk.layers.Layer):
# #         def call(self, inputs, training = None, mask = None):
# #             mu, rho = tf.split(inputs, num_or_size_splits = 2, axis = 1)
# #             sd = tf.math.log(1 + tf.math.exp(rho))
# #             batch_size = tf.shape(mu)[0]
# #             dim_z = tf.shape(mu)[1]
# #             if training:
# #                 r_sample = mu[0] + sd[0] * tf.random.normal(shape = (batch_size, dim_z))
# #                 theta_sample = mu[1] + sd[1] * tf.random.normal(shape = (batch_size, dim_z))
# #             else:
# #                 z_sample = mu[0]
# #                 theta_sample = mu[1]
# #             z_sample = tf.concat(
# #                 [r_sample * tf.math.cos(theta_sample), r_sample * tf.math.sin(theta_sample)],
# #                 axis = 1
# #             )
# #
# #             z_params = {'mu': mu, 'sd': sd}
# #             return z_sample, z_params
# #
# #     def __init__(self, dim_z):
# #         self.dim_z = dim_z
# #         self.mu_prior = [1.0]
# #         self.sigma_prior = [1.0]
# #
# #     def create_params_layer(self, name = 'z_params'):
# #         return tfk.layers.Dense(2 * 2, activation = None, name = name)
# #
# #     def create_sampler_layer(self):
# #         return self.Sampler()
# #
# #     def kl_divergence(self, data):
# #         z_sample, z_params = data
# #         mu = z_params['mu']; sd = z_params['sd']
# #         mu_targets =
# #         result = - 0.5 * tf.math.reduce_sum(1 + tf.math.log(tf.math.square(sd)) -
# #             tf.math.square(mu) - tf.math.square(sd), axis = 1)
# #         #print("kl_divergence(raw) shape: ", kl_divergence.shape)
# #         return tf.math.reduce_mean(result)
#
#
# class VAE_Encoder(tfk.layers.Layer):
#     def __init__(self, latent_model_class, dim_x, dim_z, name = "encoder"):
#         super().__init__(name = name)
#         self.dim_z = dim_z
#         self.dim_x = dim_x
#         self.latent_model_class = latent_model_class
#
#     def build(self, input_shape):
#         # self.input = tfk.layers.InputLayer(input_shape = input_shape)
#         self.dense1 = tfk.layers.Dense(100, activation = 'relu')
#         #num_dist_params = 2 * self.dim_z
#         self.z_dist_model = self.latent_model_class(self.dim_z)
#         self.z_params = self.z_dist_model.create_params_layer()
#         # self.z_mu = tfk.layers.Dense(self.dim_z, activation = None, name = 'z_mu')
#         # self.z_sigma_raw = tfk.layers.Dense(self.dim_z, activation = None, name = 'z_sigma_raw')
#         self.sampler_z = self.z_dist_model.create_sampler_layer()
#
#     def call(self, inputs, training = None, mask = None):
#         #x = self.dense1(inputs)
#         x = inputs
#         z_params = self.z_params(x)
#         z_sample, z_params = self.sampler_z(z_params)
#         # z_mu = self.z_mu(x)
#         # z_sigma_raw = self.z_sigma_raw(x)
#         # z_sample, z_sigma = self.sampler_z([z_mu, z_sigma_raw])
#         return z_sample, z_params
#
# class VAE_Decoder(tfk.layers.Layer):
#     def __init__(self, dim_x, dim_z, name = "decoder"):
#         super().__init__(name = name)
#         self.dim_z = dim_z
#         self.dim_x = dim_x
#
#     def build(self, input_shape):
#         self.dense1 = tfk.layers.Dense(100, activation = 'relu')
#         self.dense2 = tfk.layers.Dense(self.dim_x, activation = None)
#
#     def call(self, inputs, training = None, mask = None):
#         #x = self.dense1(inputs)
#         x = inputs
#         x = self.dense2(x)
#         return x
#
# class VariationalAutoencoderModel(tfk.Model):
#     def __init__(self, latent_model_class, dim_x, dim_z, kl_weight, **kwargs):
#         super().__init__(**kwargs)
#         self.dim_x = dim_x
#         self.dim_z = dim_z
#         self.kl_weight = kl_weight
#         self.encoder = VAE_Encoder(latent_model_class = latent_model_class,
#             dim_x = self.dim_x, dim_z = self.dim_z)
#         self.decoder = VAE_Decoder(dim_x = self.dim_x, dim_z = self.dim_z)
#
#     def call(self, inputs, training = None, mask = None):
#         x_true = inputs
#         z_sample, z_params = self.encoder(x_true)
#         x_reconstr = self.decoder(z_sample)
#         kl_divergence = self.encoder.z_dist_model.kl_divergence([z_sample, z_params])
#         reconstr_error = tf.reduce_sum(tf.math.square(x_reconstr - x_true), axis = 1)
#         #reconstr_error = -tfk.losses.CosineSimilarity(axis = 1)(x_reconstr, x_true)
#         reconstr_error = tf.math.reduce_mean(reconstr_error)
#             #tf.keras.losses.mean_squared_error(x_reconstr, x_true, axis = 1)
#
#
#
#         self.add_loss(
#             reconstr_error + self.kl_weight * kl_divergence)
#         self.add_metric(reconstr_error, name = 'reconstr_error')
#         self.add_metric(kl_divergence, name = 'kl_divergence')
#         return x_reconstr, z_params
#
#     # def train_step(self, data):
#     #     x_true = data
#     #     with tf.GradientTape() as tape:
#     #         mu, sd, x_reconstr = self(x_true, training = True)
#     #         # reconstr_loss = self._reconstr_loss(x_true, x_reconstr)
#     #         # kl_loss = tf.math.reduce_sum(self.losses)  # vae.losses is a list
#     #         # total_vae_loss = reconstr_loss + kl_loss
#     #         total_vae_loss = tf.math.reduce_sum(self.losses)
#     #     # Compute and apply gradients
#     #     gradients = tape.gradient(total_vae_loss, self.trainable_variables)
#     #     self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
#     #     # Update metrics (includes the metric that tracks the loss)
#     #     self.compiled_metrics.update_state(x_true, x_reconstr)
#     #     # Return a dict mapping metric names to current value
#     #     return {m.name: m.result() for m in self.metrics}
#
# class VariationalAutoencoder2(object):
#     def __init__(self, latent_model_class, dim_x, dim_z = 2, kl_weight = 1.0, batch_size = 32, learning_rate = 0.0001):
#         self.model = VariationalAutoencoderModel(latent_model_class , dim_x, dim_z, kl_weight)
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.model.compile(
#             loss = None,
#             optimizer = tfk.optimizers.Adam(self.learning_rate),
#             run_eagerly = True
#         )
#
#     def train(self, data, epochs = 10, verbose = False):
#         logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#         tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq = 1)
#         rand_indices = np.arange(data.shape[0])
#         np.random.shuffle(rand_indices)
#         self.training_history = self.model.fit(
#             x = data[rand_indices, :], y = None,
#             shuffle = True, epochs = epochs,
#             batch_size = self.batch_size,
#             validation_split = 0.3, verbose = verbose,
#             callbacks = [tensorboard_callback]
#         )
#         self.x_reconstr, self.z_params = self.model.predict(data)
#         return self.training_history
#
#     def plot_losses(self):
#         metrics = ['loss', 'reconstr_error', 'kl_divergence']
#         fig, axes = plt.subplots(1, len(metrics), figsize = (15, 5))
#         history =  self.training_history.history
#         for i, metric in enumerate(metrics):
#             axes[i].semilogy(history[metric], label = 'training')
#             axes[i].semilogy(history['val_' + metric], label = 'validation')
#             axes[i].legend()
#             axes[i].set_title(metric)

######################################################################################
# class VariationalAutoencoderBase(object):
#     def get_or_create(self, attr, create):
#         if not hasattr(self, attr):
#             setattr(self, attr, create())
#         return getattr(self, attr)
#
#     def get_encoder(self):
#         return self.get_or_create('encoder', self.build_encoder)
#
#     def get_decoder(self):
#         return self.get_or_create('decoder', self.build_decoder)
#

class EpochProgressCallback(tfk.callbacks.Callback):
    def __init__(self, total_num_epochs):
        self.pbar = tqdm(total = total_num_epochs)
    def on_epoch_begin(self, epoch, logs=None):
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        self.pbar.close()


# class NegativeBinomialLayer(tfk.layers.Layer):
#     def __init__(self):
#         super().__init__()
#
#
#     def make_distribution(self, inputs):
#         total_count, p_logit = inputs
#         total_count = tf.math.floor(1. + tf.math.exp(total_count))
#         dist = tfp.distributions.Independent(
#             tfp.distributions.NegativeBinomial(
#                 total_count, logits = p_logit, validate_args = True, name='NegativeBinomial'
#             ), reinterpreted_batch_ndims = 1
#         )
#         return dist
#
#     def build(self, input_shape):
#         self.x_output_rv = tfp.layers.DistributionLambda(
#             make_distribution_fn = self.make_distribution,
#             convert_to_tensor_fn = tfp.distributions.Distribution.sample
#         )
#         self.p_logit = tf.Variable(0.1, name = 'capture_rate')
#
#     def call(self, inputs, training = None, mask = None):
#         x_in, x_out_params = inputs
#         predicted = self.x_output_rv([x_out_params, self.p_logit])
#         neg_log_prob = -predicted.log_prob(tf.math.floor(tf.math.exp(x_in) - 1))
#         neg_log_prob = tf.reduce_mean(neg_log_prob)
#         self.add_loss(neg_log_prob)
#         self.add_metric(neg_log_prob, name = 'x_neg_log_prob')
#         return x_out_params

# Using  https://github.com/theislab/dca/blob/master/dca/loss.py
class NegativeBinomialContinuousDistribution(tfp.distributions.distribution.Distribution):
    def __init__(self, mu, theta, validate_args = False, allow_nan_stats = True, name = 'NegativeBinomial'):
        parameters = {'mu': mu, 'theta': theta}
        super().__init__(
            dtype = tf.float32,
            reparameterization_type = tfp.distributions.FULLY_REPARAMETERIZED,
            validate_args = validate_args,
            allow_nan_stats = allow_nan_stats,
            parameters = parameters,
            name = name
        )

    def _sample_n(self, n, seed = None):
        mu = self._parameters['mu']
        theta = self._parameters['theta']
        p = theta / (mu + theta)
        output_shape = mu.shape.as_list()
        output_shape.insert(0, n)
        # print(locals())
        if (n > 1):
            raise NotImplementedError("sample_n for n > 1")
        return np.random.negative_binomial(theta, p, size = output_shape)

    def log_prob(self, x):
        eps = 1e-6
        mu = self._parameters['mu']
        theta = self._parameters['theta']
        theta = tf.minimum(theta, 1e6)
        t1 = tf.math.lgamma(x + theta + eps) - tf.math.lgamma(theta + eps) - tf.math.lgamma(x + 1.0)
        t2 = - (theta + x) * tf.math.log(1.0 + (mu / (theta + eps))) + x * (tf.math.log(mu + eps) - tf.math.log(theta + eps))
        return t1 + t2


class IndependentNegativeBinomial(tfp.layers.DistributionLambda):
    def __init__(self, validate_args = True, convert_to_tensor_fn = tfp.distributions.Distribution.sample):
        super().__init__(
            make_distribution_fn = lambda t : IndependentNegativeBinomial.new(t, validate_args),
            convert_to_tensor_fn = convert_to_tensor_fn
        )

    @staticmethod
    def new(params, validate_args = False, name = None):
        with tf.name_scope(name or 'IndependentNegativeBinomial'):
            mu, theta = tf.split(params, 2, axis = -1)
            mu = tf.math.softplus(mu)
            theta = tf.math.softplus(theta)

            dist = tfp.distributions.Independent(
                NegativeBinomialContinuousDistribution(
                    mu = mu, theta = theta, validate_args = validate_args
                ),
                reinterpreted_batch_ndims = 1
            )
            return dist

    # def loss(self, y_true, y_pred):
    #     t1 = tf.math.lgamma(theta+eps) + tf.math.lgamma(y_true+1.0) - tf.math.lgamma(y_true+theta+eps)
    #     t2 = (theta+y_true) * tf.math.log(1.0 + (y_pred/(theta+eps))) + (y_true * (tf.math.log(theta+eps) - tf.math.log(y_pred+eps)))


    # def build(self, input_shape):
    #     self.p_logit = tf.Variable(0.1, name = 'capture_rate')
    #     super().build(input_shape)

    # def call(self, inputs, training = None, mask = None):
    #     return super(IndependentNegativeBinomial, self).call([inputs, self.p_logit], training, mask)


class VariationalAutoencoder_TFP(tfk.Model):
    def __init__(self,
            dim_x, dim_z = 2,
            kl_weight = 1.0, batch_size = 128, learning_rate = 0.01
        ):
        super().__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.kl_weight = kl_weight
        self.batch_size = batch_size
        self.learning_rate = learning_rate


    def build(self, input_shape):
        # print('VariationalAutoencoder_TFP/build input_shape = ', input_shape)
        self.build_encoder()
        self.build_decoder()

    def build_encoder(self):
        self.enc_dense1 = tfk.layers.Dense(100, activation = 'relu')
        num_dist_params = self.dim_z
        # Set prior
        prior = tfp.distributions.Independent(tfp.distributions.Normal(
            loc = tf.zeros(self.dim_z), scale = 1.),
            reinterpreted_batch_ndims = 1
        )
        # Model class
        z_model_class = tfp.layers.IndependentNormal
        self.z_dist_params = tfk.layers.Dense(4, activation = None, name = 'z_dist_params')
        self.z_sample = z_model_class(
            self.dim_z,
            convert_to_tensor_fn = tfp.distributions.Distribution.sample,
            activity_regularizer = tfp.layers.KLDivergenceRegularizer(prior, weight = self.kl_weight),
            name='z_sample'
        )

        # self.z_sample[:, 0] * tf.math.cos(self.z_sample[:, 1])
        # self.z_sample[:, 0] * tf.math.cos(self.z_sample[:, 1])

    def build_decoder(self):
        self.dec_dense1 = tfk.layers.Dense(100, activation = 'relu')
        self.x_dist_params = tfk.layers.Dense(2 * self.dim_x, activation = None, name = 'x_dist_params')
        self.x_dist_layer = IndependentNegativeBinomial()



    def call(self, inputs, training = None, mask = None):
        x_in, x_in_2, scaling_factors = inputs
        z1 = self.z_dist_params(self.enc_dense1(x_in))
        #z2 = self.z_dist_params(self.enc_dense1(x_in_2))
        #z_params = tf.stack([z1[:, 0], z2[:, 0], z1[:, 1], z2[:, 1]], axis = 1)
        z_params = z1
        z_mu, z_sigma = tf.split(z_params, 2, axis = -1)
        z_mu_norm = tf.norm(z_mu, axis = -1)
        z_mu_norm_loss = 0.1 * tf.reduce_mean(tf.math.square(z_mu_norm - 1.0))
        self.add_loss(z_mu_norm_loss)
        self.add_metric(z_mu_norm_loss, name = 'z_mu_norm_loss')

        z_sample = self.z_sample(z_params)

        x_params = self.x_dist_params(self.dec_dense1(z_sample))
        x_dist = self.x_dist_layer(x_params)
        # print(x_dist)
        # print(self.x_dist_layer.p_logit)
        neg_log_prob = -x_dist.log_prob(scaling_factors * (tf.math.exp(x_in) - 1))
        # print(neg_log_prob)
        neg_log_prob = tf.reduce_mean(neg_log_prob)
        self.add_loss(neg_log_prob)
        self.add_metric(neg_log_prob, name = 'x_neg_log_prob')

        #kl_divergence = self.encoder.z_dist_model.kl_divergence([z_sample, z_params])
        # reconstr_error = tf.reduce_sum(tf.math.square(x_reconstr - x_in), axis = 1)
        # reconstr_error = tf.math.reduce_mean(reconstr_error)
        #
        # self.add_loss(reconstr_error)
        # self.add_metric(reconstr_error, name = 'reconstr_error')
        #self.add_metric(kl_divergence, name = 'kl_divergence')

        return x_params, z_params

    def train(self, data, scaling_factors = None, epochs = 10, verbose = False):
        if scaling_factors is None:
            scaling_factors = np.ones(data.shape[0])
        self.compile(
            loss = None,
            optimizer = tfk.optimizers.Adam(self.learning_rate),
            run_eagerly = True
        )
        x1, x2 = data
        rand_indices = np.arange(x1.shape[0])
        np.random.shuffle(rand_indices)
        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        self.training_history = self.fit(
            x = [x1[rand_indices, :], x2[rand_indices, :], scaling_factors[rand_indices]], y = None,
            shuffle = True, epochs = epochs,
            batch_size = self.batch_size,
            validation_split = 0.2, verbose = verbose,
            callbacks = [tensorboard_callback, EpochProgressCallback(epochs)]
        )
        x_params, z_params = self.predict([x1, x2, scaling_factors])
        mu, sigma = tf.split(z_params, 2, axis = -1)
        # total_count, p_count_logit = tf.split(x_params, 2, axis = -1)
        self.z_params = {
            'mu': mu.numpy(),
            'sigma': tf.math.softplus(sigma).numpy()
        }
        mu, theta = tf.split(x_params, 2, axis = -1)
        self.x_params = {
            'mu': tf.math.softplus(mu).numpy(),
            'theta': tf.math.softplus(theta).numpy()
        }
        return self.training_history

    def plot_losses(self):
        metrics = ['loss', 'kl_divergence'] #, 'kl_divergence'
        fig, axes = plt.subplots(1, len(metrics), figsize = (15, 5))
        history =  self.training_history.history
        for i, metric in enumerate(metrics):
            axes[i].semilogy(history[metric], label = 'training')
            axes[i].semilogy(history['val_' + metric], label = 'validation')
            axes[i].legend()
            axes[i].set_title(metric)

# class KoopmanAutoencoder(object):
#     def __init__(self, n_var, latent_dim = 2, batch_size = 32):
#         super().__init__()
#         self.n_var = n_var
#         self.latent_dim = latent_dim
#         self.batch_size = batch_size
#
#         self.model = tfk.Model(self.encoder_input)
#
#     def build_encoder(self):
#         self.encoder_input = tfk.Input(
#             shape = (self.n_var,), name = 'encoder_input')
#         x1 = tfk.layers.Dense(self.latent_dim)(self.encoder_input)
#         self.encoder_output = x1
#         self.encoder = tfk.layers.Model(self.encoder_input, self.encoder_output)
#         return encoder
#
#     def build_decoder(self):
#         self.decoder_input = tfk.Input(
#             shape = (self.latent_dim,), name = 'decoder_input')
#         x1 = tfk.layers.Dense(self.n_var)(self.decoder_input)
#         self.decoder_output = x1
#         self.decoder = tfk.layers.Model(self.decoder_input, self.decoder_output)
#         return
#
#     def train(self, x_in):
#         pass
