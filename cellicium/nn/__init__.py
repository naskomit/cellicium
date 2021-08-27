import datetime, os, time
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp
from tqdm import tqdm
import matplotlib.pyplot as plt

class NormalDistributionVariationalLayer(tfk.layers.Layer):
    def __init__(self, reconstr_weight = 1.0, kl_weight = 1.0):
        super().__init__()
        self.reconstr_weight = reconstr_weight
        self.kl_weight = kl_weight

    def build(self, inputs):
        print("Inputs")
        print(inputs)
        x_in = tfk.Input(shape = inputs[0].as_list()[-1])
        x_out = tfk.Input(shape = inputs[1].as_list()[-1])
        z_mean = tfk.Input(shape = inputs[2].as_list()[-1])
        z_log_var = tfk.Input(shape = inputs[3].as_list()[-1])
        reconstr_loss = self.reconstr_weight * tf.losses.mean_squared_error(x_in, x_out)
        kl_loss = - 0.5 * self.kl_weight * tf.math.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis = -1
        )
        loss = tf.reduce_mean(reconstr_loss + kl_loss)
        self.add_loss(loss, inputs = inputs)
        self.add_metric(tfk.metrics.MeanSquaredError()(x_in, x_out), name = 'reconstr_loss_1')
        self.model = tfk.Model([x_in, x_out, z_mean, z_log_var], loss)
        return self.model

    # def call(self, inputs):
    #     x_in = inputs[0]
    #     x_out = inputs[1]
    #     loss = self.model(inputs)
        # z_mean = inputs[2]
        # z_log_var = inputs[3]
        # reconstr_loss = self.reconstr_weight * tf.losses.mean_squared_error(x_in, x_out)
        # kl_loss = - 0.5 * self.kl_weight * tf.math.reduce_sum(
        #     1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
        #     axis = -1
        # )
        # loss = tf.reduce_mean(reconstr_loss + kl_loss)
        # self.add_loss(loss, inputs = inputs)
        # self.add_metric(tfk.metrics.MeanSquaredError()(x_in, x_out), name = 'reconstr_loss_1')
#        return x_out


class VariationalAutoencoder1(object):
    def __init__(self, n_var, latent_dim = 2, batch_size = 16):
        super().__init__()
        self.n_var = n_var
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.build()


    def build(self):
        # Input
        self.encoder_input = tfk.Input(shape = (self.n_var,), name = 'encoder_input')
        # Encoder
        x = tfk.layers.Dense(32, activation = 'relu', name = 'encoder_1')(self.encoder_input)
        z_mean = tfk.layers.Dense(self.latent_dim, name = 'encoder_mean')(x)
        z_log_var = tfk.layers.Dense(self.latent_dim, name = 'encoder_variance')(x)
        self.encoder = tfk.Model(self.encoder_input, [z_mean, z_log_var])
        # Sampler
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.random.normal(
                shape = (tf.shape(z_mean)[0], self.latent_dim),
                mean = 0.0, stddev = 1.0)
            return z_mean + tf.exp(z_log_var) * epsilon
        z = tfk.layers.Lambda(sampling)([z_mean, z_log_var])
        # Decoder
        self.decoder_input = tfk.Input(z.shape[1])
        self.decoder_layer = tfk.layers.Dense(self.n_var, activation = 'relu', name = 'decoder_1')
        #self.time_step_layer = tfk.layers.Dense(self.latent_dim)
        x_reconstr = self.decoder_layer(self.decoder_input)
        #x_next = self.decoder_layer(self.time_step_layer(self.decoder_input))
        self.decoder = tfk.Model(self.decoder_input, x_reconstr)
        reconstr_output = self.decoder(z)

        self.loss_output = NormalDistributionVariationalLayer()(
            [self.encoder_input, reconstr_output, z_mean, z_log_var]
        )

        self.model = tfk.Model(self.encoder_input, [self.loss_output])

    def train(self, x_in):

        self.model.compile(
            optimizer = 'rmsprop',
            loss = None
#             metrics = ['cosine_similarity', 'mean_squared_error', 'kullback_leibler_divergence']
        )
        self.model.summary()
        fit_result = self.model.fit(
            x = x_in, y = None,
            shuffle = True, epochs = 100,
            batch_size = self.batch_size,
            validation_split = 0.3, verbose = True)
        return fit_result



######################################################################################

class GaussianModel():
    class Sampler(tfk.layers.Layer):
        def call(self, inputs, training = None, mask = None):
            mu, rho = tf.split(inputs, num_or_size_splits = 2, axis = 1)
            sd = tf.math.log(1 + tf.math.exp(rho))
            batch_size = tf.shape(mu)[0]
            dim_z = tf.shape(mu)[1]
            if training:
                z_sample = mu + sd * tf.random.normal(shape = (batch_size, dim_z))
            else:
                z_sample = mu
            z_params = {'mu': mu, 'sd': sd}
            return z_sample, z_params

    def __init__(self, dim_z):
        self.dim_z = dim_z

    def create_params_layer(self, name = 'z_params'):
        return tfk.layers.Dense(2 * self.dim_z, activation = None, name = name)

    def create_sampler_layer(self):
        return self.Sampler()

    def kl_divergence(self, data):
        z_sample, z_params = data
        mu = z_params['mu']; sd = z_params['sd']
        result = - 0.5 * tf.math.reduce_sum(1 + tf.math.log(tf.math.square(sd)) -
            tf.math.square(mu) - tf.math.square(sd), axis = 1)
        #print("kl_divergence(raw) shape: ", kl_divergence.shape)
        return tf.math.reduce_mean(result)

class VAE_Encoder(tfk.layers.Layer):
    def __init__(self, dim_x, dim_z, name = "encoder"):
        super().__init__(name = name)
        self.dim_z = dim_z
        self.dim_x = dim_x

    def build(self, input_shape):
        # self.input = tfk.layers.InputLayer(input_shape = input_shape)
        self.dense1 = tfk.layers.Dense(100, activation = 'relu')
        #num_dist_params = 2 * self.dim_z
        self.z_dist_model = GaussianModel(self.dim_z)
        self.z_params = self.z_dist_model.create_params_layer()
        # self.z_mu = tfk.layers.Dense(self.dim_z, activation = None, name = 'z_mu')
        # self.z_sigma_raw = tfk.layers.Dense(self.dim_z, activation = None, name = 'z_sigma_raw')
        self.sampler_z = self.z_dist_model.create_sampler_layer()

    def call(self, inputs, training = None, mask = None):
        #x = self.dense1(inputs)
        x = inputs
        z_params = self.z_params(x)
        z_sample, z_params = self.sampler_z(z_params)
        # z_mu = self.z_mu(x)
        # z_sigma_raw = self.z_sigma_raw(x)
        # z_sample, z_sigma = self.sampler_z([z_mu, z_sigma_raw])
        return z_sample, z_params

class VAE_Decoder(tfk.layers.Layer):
    def __init__(self, dim_x, dim_z, name = "decoder"):
        super().__init__(name = name)
        self.dim_z = dim_z
        self.dim_x = dim_x

    def build(self, input_shape):
        self.dense1 = tfk.layers.Dense(100, activation = 'relu')
        self.dense2 = tfk.layers.Dense(self.dim_x, activation = None)

    def call(self, inputs, training = None, mask = None):
        #x = self.dense1(inputs)
        x = inputs
        x = self.dense2(x)
        return x

class VariationalAutoencoderModel(tfk.Model):
    def __init__(self, dim_x, dim_z, kl_weight, **kwargs):
        super().__init__(**kwargs)
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.kl_weight = kl_weight
        self.encoder = VAE_Encoder(dim_x = self.dim_x, dim_z = self.dim_z)
        self.decoder = VAE_Decoder(dim_x = self.dim_x, dim_z = self.dim_z)

    def call(self, inputs, training = None, mask = None):
        x_true = inputs
        z_sample, z_params = self.encoder(x_true)
        x_reconstr = self.decoder(z_sample)
        kl_divergence = self.encoder.z_dist_model.kl_divergence([z_sample, z_params])

        reconstr_error = tf.math.reduce_mean(
            tf.keras.losses.mean_squared_error(x_reconstr, x_true)
        )

        self.add_loss(
            reconstr_error + self.kl_weight * kl_divergence)
        self.add_metric(reconstr_error, name = 'reconstr_error')
        self.add_metric(kl_divergence, name = 'kl_divergence')
        return x_reconstr, z_params

    # def train_step(self, data):
    #     x_true = data
    #     with tf.GradientTape() as tape:
    #         mu, sd, x_reconstr = self(x_true, training = True)
    #         # reconstr_loss = self._reconstr_loss(x_true, x_reconstr)
    #         # kl_loss = tf.math.reduce_sum(self.losses)  # vae.losses is a list
    #         # total_vae_loss = reconstr_loss + kl_loss
    #         total_vae_loss = tf.math.reduce_sum(self.losses)
    #     # Compute and apply gradients
    #     gradients = tape.gradient(total_vae_loss, self.trainable_variables)
    #     self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    #     # Update metrics (includes the metric that tracks the loss)
    #     self.compiled_metrics.update_state(x_true, x_reconstr)
    #     # Return a dict mapping metric names to current value
    #     return {m.name: m.result() for m in self.metrics}

class VariationalAutoencoder2(object):
    def __init__(self, dim_x, dim_z = 2, kl_weight = 1.0, batch_size = 32, learning_rate = 0.001):
        self.model = VariationalAutoencoderModel(dim_x, dim_z, kl_weight)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.compile(
            loss = None,
            optimizer = tfk.optimizers.Adam(self.learning_rate),
            run_eagerly = True
        )

    def train(self, data, epochs = 10, verbose = False):
        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        self.training_history = self.model.fit(
            x = data, y = None,
            shuffle = True, epochs = epochs,
            batch_size = self.batch_size,
            validation_split = 0.3, verbose = verbose,
            callbacks = [tensorboard_callback]
        )
        self.x_reconstr, self.z_params = self.model.predict(data)
        return self.training_history

    def plot_losses(self):
        metrics = ['loss', 'reconstr_error', 'kl_divergence']
        fig, axes = plt.subplots(1, len(metrics), figsize = (15, 5))
        history =  self.training_history.history
        for i, metric in enumerate(metrics):
            axes[i].semilogy(history[metric], label = 'training')
            axes[i].semilogy(history['val_' + metric], label = 'validation')
            axes[i].legend()
            axes[i].set_title(metric)

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
# class VariationalAutoencoder_TFP(VariationalAutoencoderBase):
#     def __init__(self,
#         n_var, latent_dim = 2, kl_weight = 1.0, batch_size = 32,
#         learning_rate = 0.001):
#         super().__init__()
#         self.n_var = n_var
#         self.latent_dim = latent_dim
#         self.kl_weight = kl_weight
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.build()
#
#
#     def build(self):
#         x_input = tfk.Input(shape = (self.n_var,), name = 'encoder_input')
#         z = self.get_encoder()(x_input)
#         x_output = self.get_decoder()(z)
#         self.model = tfk.Model(x_input, x_output)
#         # Negative log likelihood
#         loss = lambda x, rv_x: -rv_x.log_prob(x)
#         self.model.compile(
#             loss = loss,
#             optimizer = tfk.optimizers.Adam(self.learning_rate))
#         self.get_encoder().summary()
#         self.get_decoder().summary()
#
#     def build_encoder(self):
#         layers = [tfk.layers.InputLayer(input_shape = self.n_var)]
#         num_dist_params = tfp.layers.IndependentNormal.params_size(self.latent_dim)
#         # Compute distribution parameters
#         layers.append(tfk.layers.Dense(100, activation = 'relu'))
#         layers.append(tfk.layers.Dense(num_dist_params, activation = None, name = 'z_dist_params'))
#         # Set prior
#         prior = tfp.distributions.Independent(tfp.distributions.Normal(
#             loc = tf.zeros(self.latent_dim), scale = 1.
#             ), reinterpreted_batch_ndims = 1)
#
#         layers.append(tfp.layers.IndependentNormal(
#             self.latent_dim,
#             convert_to_tensor_fn = tfp.distributions.Distribution.sample,
#             activity_regularizer = tfp.layers.KLDivergenceRegularizer(prior, weight = self.kl_weight),
#             name='z_layer'
#         ))
#         return tfk.Sequential(layers, name = 'encoder')
#
#
#     def build_decoder(self):
#         layers = [tfk.layers.InputLayer(input_shape = self.latent_dim)]
#         layers.append(tfk.layers.Dense(100, activation = 'relu'))
#         layers.append(tfk.layers.Dense(2 * self.n_var))
#         layers.append(tfp.layers.IndependentNormal(self.n_var, name='x_layer'))
#         return tfk.Sequential(layers, name = 'decoder')
#
#
#     def train(self, data):
#         logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#         tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
#         self.training_history = self.model.fit(
#             x = data, y = data,
#             shuffle = True, epochs = 200,
#             batch_size = self.batch_size,
#             validation_split = 0.3, verbose = False,
#             callbacks = [tensorboard_callback]
#         )


class KoopmanAutoencoder(object):
    def __init__(self, n_var, latent_dim = 2, batch_size = 32):
        super().__init__()
        self.n_var = n_var
        self.latent_dim = latent_dim
        self.batch_size = batch_size

        self.model = tfk.Model(self.encoder_input)

    def build_encoder(self):
        self.encoder_input = tfk.Input(
            shape = (self.n_var,), name = 'encoder_input')
        x1 = tfk.layers.Dense(self.latent_dim)(self.encoder_input)
        self.encoder_output = x1
        self.encoder = tfk.layers.Model(self.encoder_input, self.encoder_output)
        return encoder

    def build_decoder(self):
        self.decoder_input = tfk.Input(
            shape = (self.latent_dim,), name = 'decoder_input')
        x1 = tfk.layers.Dense(self.n_var)(self.decoder_input)
        self.decoder_output = x1
        self.decoder = tfk.layers.Model(self.decoder_input, self.decoder_output)
        return

    def train(self, x_in):
        pass
