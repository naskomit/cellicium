import numpy as np
import pandas as pd
import scanpy as sc
import tensorflow as tf
import tensorflow.keras as tfk
import scipy.sparse as sparse
import typing as tp
import os
from cellicium.logging import logger as log
import cellicium.scrna as crna
from cellicium.utils import display
from cellicium.logging import logger as log
from . import nn_utils as nnu
from . import base
import sklearn.neighbors as sknn

from tensorflow.python.framework.ops import EagerTensor

#################################################
# Contrastive learning models self vs. others
#################################################

class ContrastiveLossErrorLayer(tfk.layers.Layer):
    def __init__(self):
        super().__init__()
        self.T = 1.0
        self.d0 = 1.0
        self.print_counter = 0

    def cosine_similarity(self, z) -> tf.Tensor:
        # We would like latent vectors of roughly size 1
        norm = tf.norm(z, axis = -1)
        # norm_loss = tf.square(tf.reduce_mean(norm) - 1)

        # Normalize vectors
        z_norm = z / tf.reshape(norm, [-1, 1])

        # Compute dot products (cosine similarity)
        sim = tf.matmul(z_norm, tf.transpose(z_norm))
        sim = tf.math.exp(sim / self.T)

        return sim

    def distance_similarity(self, z) -> tf.Tensor:
        norm_sq = tf.reduce_sum(z*z, axis = -1)
        norm_sq = tf.reshape(norm_sq, (-1, 1))
        dist_sq = norm_sq - 2 * tf.matmul(z, tf.transpose(z)) + tf.transpose(norm_sq)
        # Add ones to the diagonal to avoid infinities on computing 1/x
        dist_sq = dist_sq + tf.eye(tf.shape(z)[0])
        sim = self.d0 / (self.d0 + dist_sq)
        # sim = tf.math.exp(- dist_sq / self.d0)

        return sim

    def call(self, inputs, *args, **kwargs):
        n_samples = tf.shape(inputs[0])[0]
        # Concatenate all the z points
        z = tf.concat(inputs, axis = 0)

        # sim = self.cosine_similarity(z)
        sim = self.distance_similarity(z)

        # Remove diagonal
        sim = sim - tf.linalg.diag(tf.linalg.diag_part(sim))
        eps = 0.0
        denom = tf.reduce_sum(sim, axis = 1) + eps

        num1 = tf.linalg.diag_part(sim, k = n_samples)
        num2 = tf.linalg.diag_part(sim, k = -n_samples)
        num = tf.concat([num1, num2], axis = 0) + eps

        loss = -tf.math.log(num/denom)
        loss = tf.reduce_mean(loss)

        # if isinstance(sim, EagerTensor) and tf.math.reduce_any(tf.math.is_nan(loss)):
        #     tf.print('Got NaN')
        #     display(pd.DataFrame(sim.numpy()))
        #     display(pd.DataFrame(num.numpy()))
        #     display(pd.DataFrame(denom.numpy()))
        #     raise ValueError("Got NaN")

        return loss


class UnimodeEncoder(base.EncoderModel):
    def __init__(self, dim_in : int, dim_z : int, n_layers : int, splitter_dropout_rate : float = 0.2,
                 dropout_rate : float = 0.2, l1 : float = 0.0, l2 : float = 0.0):
        super().__init__()
        self.dim_in = dim_in
        self.dim_z = dim_z
        self.n_layers = n_layers
        self.splitter_dropout_rate = splitter_dropout_rate
        self.dropout_rate = dropout_rate
        self.l1 = l1
        self.l2 = l2
        self.add_loss_tracker('contr_loss')
        self.network = self.build_model()

    def build_block(self, x, relu = False):
        dense_kwargs = {}
        if self.l1 > 0 or self.l2 > 0:
            dense_kwargs['kernel_regularizer'] = tfk.regularizers.l1_l2(self.l1, self.l2)
        x = tfk.layers.Dense(units = self.dim_z, **dense_kwargs)(x)
        # x = nnu.LinLogLayer(shape = self.dim_z, log_offset = 1.0)(x)
        x = tfk.layers.BatchNormalization()(x)
        # if relu:
        x = tfk.layers.Lambda(lambda y : tf.math.softplus(y))(x)
        # x = tfk.layers.Dropout(self.dropout_rate)(x)
        return x

    def build_model(self, **kwargs) -> tfk.Model:
        encoder_input = tfk.Input(self.dim_in, name = 'encoder input')
        print(f'Building model with {self.n_layers} layers and {self.dim_z} latent dimensions')
        x = encoder_input
        for i in range(self.n_layers):
            x = self.build_block(x, relu = True)

        # if i < self.n_layers - 1:
            #     x = self.build_block(x, relu = True)
            # else:
            #     x = self.build_block(x, relu = False)
        z = x
        encoder = tfk.Model(inputs = encoder_input, outputs = z, name = 'Encoder')

        x1_input = tfk.Input(self.dim_in, name = 'input x1')
        x2_input = tfk.Input(self.dim_in, name = 'input x2')

        z1 = encoder(x1_input)
        z2 = encoder(x2_input)

        contrastive_loss = ContrastiveLossErrorLayer()((z1, z2))

        self.encoder = encoder
        model = tfk.Model(inputs = [x1_input, x2_input],
                          outputs = {'contr_loss': contrastive_loss},
                          name = 'UnimodeEncoder')

        return model

    # def transform_input(self, data):
    #     mask1 = tf.cast(tf.greater(
    #         tf.random.uniform(tf.shape(data), minval=0.0, maxval = 1.0),
    #         self.splitter_dropout_rate),
    #         tf.float32)
    #     mask2 = tf.cast(tf.greater(
    #         tf.random.uniform(tf.shape(data), minval=0.0, maxval = 1.0),
    #         self.splitter_dropout_rate),
    #         tf.float32)
    #     # tf.print('dropout: ', self.dropout_rate)
    #     # tf.print(tf.reduce_sum(tf.cast(mask1 == mask2, tf.float32)))
    #     return [tf.multiply(mask1, data), tf.multiply(mask2, data)]
    #
    # def train_step(self, data):
    #     data = self.transform_input(data)
    #     return super().train_step(data)
    #
    # def test_step(self, data):
    #     data = self.transform_input(data)
    #     return super().train_step(data)


# def random_apply(func, p, x):
#     """Randomly apply function func to x with probability p."""
#     return tf.cond(
#         tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32),
#                 tf.cast(p, tf.float32)),
#         lambda: func(x),
#         lambda: x)

class ADT_Preprocessor(base.ModelManagerBase):
    def __init__(self, dim_z, n_layers, splitter_dropout_rate = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.dim_z = dim_z
        self.n_vars = None
        self.n_layers = n_layers
        self.splitter_dropout_rate = splitter_dropout_rate

    def save_encoder(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
        filepath = os.path.join(dir, f'ADT_preproc_encoder.h5')
        self.model.encoder.save_weights(filepath, save_format = 'hdf5')

    def load_encoder(self, dir, n_vars):
        self.model = UnimodeEncoder(dim_in = n_vars, dim_z = self.dim_z, n_layers = self.n_layers,
                               splitter_dropout_rate = self.splitter_dropout_rate)
        filepath = os.path.join(dir, f'ADT_preproc_encoder.h5')
        self.model.encoder.load_weights(filepath)

    def build_model(self, **kwargs) -> tfk.Model:
        n_vars = kwargs.pop('n_vars')
        model = UnimodeEncoder(dim_in = n_vars, dim_z = self.dim_z, n_layers = self.n_layers,
                               splitter_dropout_rate = self.splitter_dropout_rate)
        return model

    def train(self, adata : sc.AnnData, **kwargs):
        X = adata.X
        if sparse.issparse(X):
            X = X.todense()

        mod_dataset = tf.data.Dataset.from_tensor_slices(X.astype('float32'))

        @tf.function
        def map_fn(x):
            mask1 = tf.cast(tf.greater(tf.random.uniform(tf.shape(x), maxval = 1.0), self.splitter_dropout_rate), tf.float32)
            mask2 = tf.cast(tf.greater(tf.random.uniform(tf.shape(x), maxval = 1.0), self.splitter_dropout_rate), tf.float32)
            # tf.print('dropout: ', self.splitter_dropout_rate)
            # tf.print(tf.reduce_sum(tf.cast(mask1 == mask2, tf.float32)))
            return [tf.multiply(mask1, x), tf.multiply(mask2, x)]

        mod_dataset = mod_dataset.map(map_fn, num_parallel_calls = 4)
        kwargs['n_vars'] = adata.n_vars
        self.do_train(mod_dataset, adata.n_obs, **kwargs)

    def latent_representation(self, adata : sc.AnnData):
        X = adata.X
        X = X.todense() if sparse.issparse(X) else X
        Z = self.model.encoder.predict(X)
        result = sc.AnnData(X = Z, obs = adata.obs)
        return result


class GeneCLRModel(base.EncoderModel):
    def __init__(self, dim_in1, dim_in2, dim_z):
        super().__init__()
        self.dim_in1 = dim_in1
        self.dim_in2 = dim_in2
        self.dim_z = dim_z
        self.add_loss_tracker('mod1_reconstr_loss')
        self.add_loss_tracker('mod2_reconstr_loss')
        self.add_loss_tracker('contr_loss')
        self.network = self.build_model()

    def assemble_encoder(self, input, dropout_rate = 0.3, log_offset = 1.0, l1_reg = 0.000, l2_reg = 0.000):

        dense_kwargs = {
            # 'kernel_regularizer': tfk.regularizers.l1_l2(0.00, 0.00)
        }
        x = input
        # x = nnu.LinLogLayer(shape = self.dim_z, log_offset = log_offset)(x)
        x = tfk.layers.Dense(units = self.dim_z, **dense_kwargs)(x)
        x = tfk.layers.BatchNormalization()(x)
        # x = tfk.layers.ReLU()(x)
        x = tfk.layers.Dropout(dropout_rate)(x)

        # x = tfk.layers.Dense(units = self.dim_z, **dense_kwargs)(x)
        # x = tfk.layers.BatchNormalization()(x)
        # x = tfk.layers.ReLU()(x)
        # x = tfk.layers.Dropout(dropout_rate)(x)
        #
        # x = tfk.layers.Dense(units = self.dim_z, **dense_kwargs)(x)
        # x = tfk.layers.BatchNormalization()(x)
        # x = tfk.layers.Dropout(dropout_rate)(x)

        z = x

        model = tfk.Model(input, z)
        return model

    def assemble_decoder(self, input, output_dim, dropout_rate = 0.3, log_offset = 0.1):
        x = input
        x = tfk.layers.Dense(units = output_dim)(x) #, log_offset = log_offset
        # x = tfk.layers.BatchNormalization()(x)
        z = tfk.layers.Dropout(dropout_rate)(x)

        model = tfk.Model(input, z)
        return model

    def build_model(self) -> tfk.Model:
        input_enc1 = tfk.Input(self.dim_in1, name = 'mod1 encoder input')
        encoder1 = self.assemble_encoder(input_enc1)

        input_enc2 = tfk.Input(self.dim_in2, name = 'mod2 encoder input')
        encoder2 = self.assemble_encoder(input_enc2)

        input_dec1 = tfk.Input(self.dim_z, name = 'mod1 decoder input')
        decoder1 = self.assemble_decoder(input_dec1, self.dim_in1)

        input_dec2 = tfk.Input(self.dim_z, name = 'mod2 decoder input')
        decoder2 = self.assemble_decoder(input_dec2, self.dim_in2)

        x1_input = tfk.Input(self.dim_in1, name = 'mod1 input')
        x2_input = tfk.Input(self.dim_in2, name = 'mod2 input')

        z1 = encoder1(x1_input)
        z2 = encoder2(x2_input)

        x1_out = decoder1(z1)
        x2_out = decoder2(z2)

        cl_loss_layer = ContrastiveLossErrorLayer()
        mod1_reconstr_loss_layer = tfk.layers.Lambda(lambda x: tf.reduce_mean(tfk.losses.mean_squared_error(x[0], x[1])))
        mod2_reconstr_loss_layer = tfk.layers.Lambda(lambda x: tf.reduce_mean(tfk.losses.mean_squared_error(x[0], x[1])))

        contr_loss = cl_loss_layer((z1, z2))
        mod1_reconstr_loss = mod1_reconstr_loss_layer((x1_input, x1_out))
        mod2_reconstr_loss = mod2_reconstr_loss_layer((x2_input, x2_out))

        network = tfk.Model(inputs = (x1_input, x2_input),
                            outputs = [{'mod1_reconstr_loss': mod1_reconstr_loss,
                                        'mod2_reconstr_loss': mod2_reconstr_loss,
                                        'contr_loss': contr_loss},
                                       [x1_out, x2_out, z1, z2]],
                            name = 'GeneCLRNetwork')
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        return network


class MultimodalManager(base.ModelManagerBase):
    def __init__(self, dim_z : int = 100, **kwargs):
        super().__init__(**kwargs)
        self.dim_z = dim_z
        self.modalities = {}
        self.modality_encoders = None
        self.modality_decoders = None

    def save_encoders(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
        for k,v in self.modality_encoders.items():
            v.save_weights(os.path.join(dir, f'{k}_encoder.h5'), save_format = 'hdf5')
        for k,v in self.modality_decoders.items():
            v.save_weights(os.path.join(dir, f'{k}_decoder.h5'), save_format = 'hdf5')

    def load_encoders(self, dir, dims_dict):
        modalities = list(dims_dict.keys())
        if self.model is None:
            self.model = GeneCLRModel(
                dim_in1 = dims_dict[modalities[0]], dim_in2 = dims_dict[modalities[1]], dim_z = self.dim_z)
        self.modality_encoders = {}
        self.modality_decoders = {}
        for modality, model in zip(modalities, [self.model.encoder1, self.model.encoder2]):
            self.modality_encoders[modality] = model
            filepath = os.path.join(dir, modality + '.h5')
            model.load_weights(filepath)
            log.info(f'Loaded encoder for {modality} from {filepath}')

    def find_encoder(self, modality) -> tfk.Model:
        if modality in self.modality_encoders:
            return self.modality_encoders[modality]
        else:
            raise ValueError(f'No encoder found for modality {modality}')

    def find_decoder(self, modality) -> tfk.Model:
        if modality in self.modality_decoders:
            return self.modality_decoders[modality]
        else:
            raise ValueError(f'No decoder found for modality {modality}')

    def build_model(self, **kwargs) -> tfk.Model:
        mod1_nvars = kwargs.pop('mod1_nvars')
        mod2_nvars = kwargs.pop('mod2_nvars')

        model = GeneCLRModel(
            dim_in1 = mod1_nvars, dim_in2 = mod2_nvars,
            dim_z = self.dim_z)
        modality1 = kwargs.pop('modality1')
        modality2 = kwargs.pop('modality2')
        self.modality_encoders = {
            modality1: model.encoder1, modality2: model.encoder2
        }
        self.modality_decoders = {
            modality1: model.decoder1, modality2: model.decoder2
        }
        return model



    def train(self, mod1_data : sc.AnnData, mod2_data : sc.AnnData, **kwargs):
        # tf.random.set_seed(self.seed)
        # Create the model
        # Create the combined dataset
        n_samples = mod1_data.n_obs
        X1 = mod1_data.X
        X2 = mod2_data.X
        if sparse.issparse(X1):
            X1 = X1.todense()
        if sparse.issparse(X2):
            X2 = X2.todense()
        mod1_dataset = tf.data.Dataset.from_tensor_slices(X1)
        mod2_dataset = tf.data.Dataset.from_tensor_slices(X2)
        combined_dataset = tf.data.Dataset.zip((mod1_dataset, mod2_dataset))

        kwargs['mod1_nvars'] = mod1_data.n_vars
        kwargs['mod2_nvars'] = mod2_data.n_vars
        kwargs['modality1'] = mod1_data.uns['modality']
        kwargs['modality2'] = mod2_data.uns['modality']
        self.do_train(combined_dataset, mod1_data.n_obs, **kwargs)

    def transform_to_common_space(self, adata_list : tp.List[sc.AnnData], unify = False, obs_fields = None):
        Z_list = []
        for adata in adata_list:
            X = adata.X.todense() if sparse.issparse(adata.X) else adata.X
            modality = adata.uns['modality']
            encoder = self.find_encoder(modality)
            Z_predict = encoder.predict(X)
            Z_list.append(Z_predict)

        if unify:
            obs_list = []
            for adata in adata_list:
                modality = adata.uns['modality']
                if obs_fields:
                    obs = adata.obs[obs_fields].copy()
                else:
                    obs = pd.DataFrame({}, index = adata.obs.index)
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


    def plot_integration_umap(self, adata_list : tp.List[sc.AnnData], color = None, obs_fields = None):
        log.info(f'Input mean norm: {[np.mean(np.linalg.norm(nnu.ensure_dense(adata.X), axis = -1)) for adata in adata_list]}', )
        adata_z = self.transform_to_common_space(adata_list, unify = True, obs_fields = obs_fields)
        log.info(f'Latent mean norm: {np.mean(np.linalg.norm(adata_z.X, axis = -1))}')
        log.info('Computing neighbors....')
        sc.pp.neighbors(adata_z)
        log.info('Computing umap')
        sc.tl.umap(adata_z)
        fg = crna.pl.figure_grid(n_col= 2, n_row= 1, figsize = (40, 20))
        if color is None:
            color = ['modality']
        if 'modality' in color:
            sc.pl.umap(adata_z, color = 'modality', ax = next(fg), s = 20, show = False)
        if 'cell_type' in color:
            sc.pl.umap(adata_z, color = 'cell_type', ax = next(fg), s = 20, show = False, legend_loc = 'on data')
        return adata_z

#################################################
# Contrastive learning models using triplets
#################################################

class DoubleTripletLossLayer(tfk.layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """
    def __init__(self, margin, metric = 'l2'):
        super().__init__()
        self.margin = margin
        log.info(f'Using {metric} metric')
        if metric == 'l2':
            self.distance_f = self.distance_2
        elif metric == 'l1':
            self.distance_f = self.distance_1
        else:
            raise ValueError('`metric` must be either l1 or l2')

    @staticmethod
    def distance_2(x1, x2):
        return tf.reduce_sum(tf.square(x1 - x2), axis = -1)

    @staticmethod
    def distance_1(x1, x2):
        return tf.reduce_sum(tf.abs(x1 - x2), axis = -1)

    def call(self, input, *args, **kwargs):
        ((anchor_z1, positive_z1, negative_z1), (anchor_z2, positive_z2, negative_z2)) = input

        loss_corresp = (self.distance_f(anchor_z1, anchor_z2) +
                        self.distance_f(positive_z1, positive_z2) +
                        self.distance_f(negative_z1, negative_z2))
        ap_distance = self.distance_f(anchor_z1, positive_z1)
        an_distance = self.distance_f(anchor_z1, negative_z1)
        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss_pos_neg = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
        loss = {'corresp_loss': tf.reduce_mean(loss_corresp), 'pos_neg_loss': tf.reduce_mean(loss_pos_neg)}
        return loss


class DoubleTripletModel(base.EncoderModel):
    def __init__(self, dim_in1, dim_in2, dim_z = 20, n_layers = 1, margin = 1.0, correspondance_coeff = 0.01, metric = 'l2'):
        super().__init__()
        self.dim_in1 = dim_in1
        self.dim_in2 = dim_in2
        self.dim_z = dim_z
        self.margin = margin
        self.correspondance_coeff = correspondance_coeff
        self.metric = metric
        self.add_loss_tracker("corresp_loss", gain = correspondance_coeff)
        self.add_loss_tracker("pos_neg_loss")
        self.dropout_rate = 0.3
        self.log_offset = 1.0
        self.n_layers = n_layers

        self.network = self.build_model()

    def assemble_block(self, input, last = False):
        x = input
        #x = tfk.layers.Dense(self.dim_z)(x)
        x = nnu.LinLogLayer(shape = self.dim_z, log_offset = self.log_offset)(x)
        x = tfk.layers.BatchNormalization()(x)
        if not last:
            x = tfk.layers.ReLU()(x)
        #     x = tfk.layers.Lambda(lambda y : tf.math.softmax(y))(x)
        # if not last:
        x = tfk.layers.Dropout(self.dropout_rate)(x)
        return x

    def assemble_encoder(self, input):
        x = input
        # x = tfk.layers.Lambda(lambda y : tf.math.softplus(y))(x)
        for i in range(self.n_layers):
            last = (i == self.n_layers - 1)
            x = self.assemble_block(x, last = last)
        z = x
        model = tfk.Model(input, z)
        return model

    def build_model(self) -> tfk.Model:
        input1 = tfk.Input(self.dim_in1, name = 'encoder1 input')
        encoder1 = self.assemble_encoder(input1)

        input2 = tfk.Input(self.dim_in2, name = 'mod2 input')
        encoder2 = self.assemble_encoder(input2)

        anchor_index = tfk.Input(name = 'anchor_index', shape = 1)
        anchor_input_mod1 = tfk.Input(name = 'anchor_mod1', shape = self.dim_in1)
        positive_input_mod1 = tfk.Input(name = 'positive_mod1', shape = self.dim_in1)
        negative_input_mod1 = tfk.Input(name = 'negative_mod1', shape = self.dim_in1)
        anchor_input_mod2 = tfk.Input(name = 'anchor_mod2', shape = self.dim_in2)
        positive_input_mod2 = tfk.Input(name = 'positive_mod2', shape = self.dim_in2)
        negative_input_mod2 = tfk.Input(name = 'negative_mod2', shape = self.dim_in2)

        anchor_z1 = encoder1(anchor_input_mod1)
        positive_z1 = encoder1(positive_input_mod1)
        negative_z1 = encoder1(negative_input_mod1)

        anchor_z2 = encoder2(anchor_input_mod2)
        positive_z2 = encoder2(positive_input_mod2)
        negative_z2 = encoder2(negative_input_mod2)

        # Add a layer computing the losses
        self.loss_layer = DoubleTripletLossLayer(margin = self.margin, metric = self.metric)
        model_losses = self.loss_layer(((anchor_z1, positive_z1, negative_z1), (anchor_z2, positive_z2, negative_z2)))

        network = tfk.Model(inputs = [anchor_index, anchor_input_mod1, positive_input_mod1, negative_input_mod1, anchor_input_mod2, positive_input_mod2, negative_input_mod2],
                            outputs = model_losses, name = 'SiameseModelNetwork')
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        return network


class DoubleTripletModelManager(base.ModelManagerBase):
    def __init__(self, dim_z : int = 20, n_layers : int = 1, **kwargs):
        super().__init__(**kwargs)
        self.dim_z = dim_z
        self.n_layers = n_layers
        self.modality_encoders : tp.Dict[str, tfk.Model] = {}
        self.modality_decoders : tp.Dict[str, tfk.Model] = {}

    def save_model(self, dir : str):
        if not os.path.exists(dir):
            os.mkdir(dir)
        for k, v in self.modality_encoders.items():
            v.save_weights(os.path.join(dir, f'{k}_encoder.h5'), save_format = 'hdf5')
        # for k, v in self.modality_decoders.items():
        #     v.save_weights(os.path.join(dir, f'{k}_decoder.h5'), save_format = 'hdf5')

    def load_model(self, dir : str, dims_dict):
        modalities = list(dims_dict.keys())
        if self.model is None:
            self.model = DoubleTripletModel(
                dim_in1 = dims_dict[modalities[0]], dim_in2 = dims_dict[modalities[1]], dim_z = self.dim_z,
                n_layers = self.n_layers)
        self.modality_encoders = {}
        for modality, model in zip(modalities, [self.model.encoder1, self.model.encoder2]):
            self.modality_encoders[modality] = model
            filepath = os.path.join(dir, modality + '_encoder.h5')
            model.load_weights(filepath)
            log.info(f'Loaded encoder for {modality} from {filepath}')

    def find_encoder(self, modality):
        if modality in self.modality_encoders:
            return self.modality_encoders[modality]
        else:
            raise ValueError(f'No encoder found for modality {modality}')

    def build_model(self, **kwargs) -> tfk.Model:
        mod1_nvars = kwargs.pop('mod1_nvars')
        mod2_nvars = kwargs.pop('mod2_nvars')

        modality1 = kwargs.pop('modality1')
        modality2 = kwargs.pop('modality2')
        model = DoubleTripletModel(
            dim_in1 = mod1_nvars, dim_in2 = mod2_nvars, dim_z = self.dim_z, n_layers = self.n_layers, **kwargs)

        self.modality_encoders = {
            modality1 : model.encoder1, modality2 : model.encoder2
        }

        return model

    def train(self, ds : base.ProblemDataset, major_modality, minor_modality, n_neighbors,
              training_plan : nnu.TrainingPlan, **kwargs):
        n_samples = ds.train_mod1.n_obs
        # n_neighbors = kwargs.pop('n_neighbors', 10)
        kwargs['mod1_nvars'] = ds.get_data('train', major_modality).n_vars
        kwargs['mod2_nvars'] = ds.get_data('train', minor_modality).n_vars
        kwargs['modality1'] = major_modality
        kwargs['modality2'] = minor_modality

        def ensure_dense(X):
            X_dense = X.todense() if sparse.issparse(X) else X
            X_dense = np.asarray(X_dense)
            return X_dense

        X1 = ensure_dense(ds.get_data('train', major_modality, sort = True).X)
        X2 = ensure_dense(ds.get_data('train', minor_modality, sort = True).X)

        log.info('Computing neighbours...')
        neighb_model = sknn.NearestNeighbors(n_neighbors = n_neighbors + 1, p = 1).fit(X1)
        __, neighbors = neighb_model.kneighbors(X = X1)
        # Remove the closest point (which is the point itsef at d = 0)
        neighbors = neighbors[:, 1:]
        log.info(f'Done computing neighbours! {neighbors.shape}')


        # rng = np.random.default_rng(seed = self.seed)

        neighbors = tf.convert_to_tensor(neighbors)
        X1 = tf.convert_to_tensor(X1)
        X2 = tf.convert_to_tensor(X2)

        # @tf.function
        def map_fn(anchor_index):
            # pos_example_id = rng.choice(n_neighbors, size = None) #tf.random.uniform((1,), maxval = n_neighbors, dtype = tf.dtypes.int32)
            # pos_example_id = neighbors[ind, pos_example_id]
            # neg_example_id = rng.choice(n_samples, size = None) #tf.random.uniform((1,), maxval = n_samples, dtype = tf.dtypes.int32)

            pos_example_id = tf.random.uniform((), maxval = n_neighbors, dtype = tf.dtypes.int32)
            pos_example_id = neighbors[anchor_index, pos_example_id]
            neg_example_id = tf.random.uniform((), maxval = n_samples, dtype = tf.dtypes.int32)

            anchor_mod1 = X1[anchor_index, :]
            anchor_mod2 = X2[anchor_index, :]
            pos_mod1 = X1[pos_example_id, :]
            pos_mod2 = X2[pos_example_id, :]
            neg_mod1 = X1[neg_example_id, :]
            neg_mod2 = X2[neg_example_id, :]

            return (anchor_index, anchor_mod1, pos_mod1, neg_mod1,
                    anchor_mod2, pos_mod2, neg_mod2)

        dataset = tf.data.Dataset.from_tensor_slices(tf.range(n_samples))
        triplet_dataset = dataset.map(map_fn, num_parallel_calls = 8, deterministic = False)

        # log.info('Creating dataset!')
        # triplet_dataset = tf.data.Dataset.from_generator(
        #     triplet_generator, output_signature = (
        #         tf.TensorSpec(shape=(), dtype=tf.int32),
        #         tf.TensorSpec(shape=(X1.shape[1], ), dtype=tf.float32),
        #         tf.TensorSpec(shape=(X1.shape[1], ), dtype=tf.float32),
        #         tf.TensorSpec(shape=(X1.shape[1], ), dtype=tf.float32),
        #         tf.TensorSpec(shape=(X2.shape[1], ), dtype=tf.float32),
        #         tf.TensorSpec(shape=(X2.shape[1], ), dtype=tf.float32),
        #         tf.TensorSpec(shape=(X2.shape[1], ), dtype=tf.float32)
        #     ))

        self.do_train(triplet_dataset, n_samples, training_plan, **kwargs)

    def transform_to_common_space(self, adata_list : tp.List[sc.AnnData], unify = False, obs_fields = None):
        Z_list = []
        for adata in adata_list:
            X = adata.X.todense() if sparse.issparse(adata.X) else adata.X
            modality = adata.uns['modality']
            Z_predict = self.find_encoder(modality).predict(X)
            Z_list.append(Z_predict)

        if unify:
            obs_list = []
            for adata in adata_list:
                modality = adata.uns['modality']
                if obs_fields:
                    obs = adata.obs[obs_fields].copy()
                else:
                    obs = pd.DataFrame({}, index = adata.obs.index)
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

    def joint_embedding(self, adata_list : tp.List[sc.AnnData]) -> sc.AnnData:
        z1_adata, z2_adata = self.transform_to_common_space(adata_list, unify = False)
        # Ensure they have the same order
        z2_adata = z2_adata[z1_adata.obs.index, :]
        z_X = (z1_adata.X + z2_adata.X) / 2.0
        z_data = sc.AnnData(X = z_X, var = z1_adata.var, obs = z1_adata.obs)
        return z_data



    def plot_integration_umap(self, adata_list : tp.List[sc.AnnData], color = None, obs_fields = None):
        adata = self.transform_to_common_space(adata_list, unify = True, obs_fields = obs_fields)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        fg = crna.pl.figure_grid(n_col= 2, n_row= 1, figsize = (40, 20))
        if color is None:
            color = ['modality']
        if 'modality' in color:
            sc.pl.umap(adata, color = 'modality', ax = next(fg), s = 20, show = False)
        if 'cell_type' in color:
            sc.pl.umap(adata, color = 'cell_type', ax = next(fg), s = 20, show = False, legend_loc = 'on data')
        return adata
