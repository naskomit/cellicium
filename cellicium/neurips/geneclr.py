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
from cellicium.logging import logger as log
from . import nn_utils as nnu
from . import base


class ContrastiveLossErrorLayer(tfk.layers.Layer):
    def __init__(self):
        super().__init__()
        self.T = 1.0
        self.d0 = 1.0

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
        sim = self.d0 / dist_sq

        return sim

    def call(self, inputs, *args, **kwargs):
        n_samples = tf.shape(inputs[0])[0]
        # Concatenate all the z points
        z = tf.concat(inputs, axis = 0)

        #sim = self.cosine_similarity(z)
        sim = self.distance_similarity(z)

        # Exponentiate and remove diagonal
        sim = sim - tf.linalg.diag(tf.linalg.diag_part(sim))

        denom = tf.reduce_sum(sim, axis = 1)

        num1 = tf.linalg.diag_part(sim, k = n_samples)
        num2 = tf.linalg.diag_part(sim, k = -n_samples)
        num = tf.concat([num1, num2], axis = 0)

        loss = tf.math.log(num / denom)
        loss = -tf.reduce_mean(loss)
        return loss


class ModeEncoder(base.EncoderModel):
    def __init__(self, dim_in : int, dim_z : int, n_layers : int,
                 dropout_rate :float = 0.2, l1 : float = 0.0, l2 : float = 0.0):
        super().__init__()
        self.dim_in = dim_in
        self.dim_z = dim_z
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.l1 = l1
        self.l2 = l2
        self.add_loss_tracker('contr_loss')

    def build_block(self, x, relu = False):
        dense_kwargs = {}
        if self.l1 > 0 or self.l2 > 0:
            dense_kwargs['kernel_regularizer'] = tfk.regularizers.l1_l2(self.l1, self.l2)
        x = tfk.layers.Dense(units = self.dim_z, **dense_kwargs)(x)
        x = tfk.layers.BatchNormalization()(x)
        if relu:
            x = tfk.layers.ReLU()(x)
        x = tfk.layers.Dropout(self.dropout_rate)(x)
        return x

    def build_model(self) -> tfk.Model:
        encoder_input = tfk.Input(self.dim_in, name = 'encoder input')
        x = encoder_input
        for i in range(self.n_layers):
            if i < self.n_layers - 1:
                x = self.build_block(x, relu = True)
            else:
                x = self.build_block(x, relu = False)
        # z = x
        contrastive_loss = ContrastiveLossErrorLayer()(x)
        return tfk.Model(encoder_input, {'contr_loss': contrastive_loss})


class UnimodalManager(base.ModelManagerBase):
    def __init__(self, dim_z, **kwargs):
        super().__init__(**kwargs)
        self.dim_z = dim_z
        self.n_vars = None

    def build_model(self, **kwargs) -> tfk.Model:
        n_vars = kwargs.pop('n_vars')

        return ModeEncoder(dim_in = n_vars, dim_z = self.dim_z, n_layers = 3)

    def train(self, adata : sc.AnnData, **kwargs):
        X = adata.X
        if sparse.issparse(X):
            X = X.todense()
        mod_dataset = tf.data.Dataset.from_tensor_slices(X)
        kwargs['n_vars'] = adata.n_vars
        self.do_train(mod_dataset, **kwargs)


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
        fg = crna.pl.figure_grid(ncol = 2, nrow = 1, figsize = (40, 20))
        if color is None:
            color = ['modality']
        if 'modality' in color:
            sc.pl.umap(adata_z, color = 'modality', ax = next(fg), s = 20, show = False)
        if 'cell_type' in color:
            sc.pl.umap(adata_z, color = 'cell_type', ax = next(fg), s = 20, show = False, legend_loc = 'on data')
        return adata_z
