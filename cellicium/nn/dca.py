import os
import pickle
import numpy as np
import scipy.sparse as sparse
import scanpy as sc
import scanpy.logging as log
import pandas as pd
import tensorflow as tf
import tensorflow.keras as tfk
import sklearn.model_selection as skms
from typing import Union

def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts = 1)
        sc.pp.filter_cells(adata, min_counts = 1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata


advanced_activations = ('PReLU', 'LeakyReLU')

# ===================================================================
# Activations
# ===================================================================
MeanAct = lambda x: tf.clip_by_value(tf.math.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.math.softplus(x), 1e-4, 1e4)

# ===================================================================
# Layers
# ===================================================================
ColwiseMultLayer = tfk.layers.Lambda(lambda l: l[0] * tf.reshape(l[1], (-1, 1)))


class SliceLayer(tfk.layers.Layer):
    def __init__(self, index, **kwargs):
        self.index = index
        super().__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('Input should be a list')

        super().build(input_shape)

    def call(self, x):
        assert isinstance(x, list), 'SliceLayer input is not a list'
        return x[self.index]

    def compute_output_shape(self, input_shape):
        return input_shape[self.index]


def _nan2zero(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

def _nan2inf(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x)+np.inf, x)

def _nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.math.is_nan(x), tf.float32))
    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)

# dispersion (theta) parameter is a scalar by default.
# scale_factor scales the nbinom mean before the
# calculation of the loss to balance the
# learning rates of theta and network weights
class NB(object):
    def __init__(self, theta=None, masking=False, scope='nbinom_loss/',
                 scale_factor=1.0, debug=False):

        # for numerical stability
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.debug = debug
        self.scope = scope
        self.masking = masking
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        with tf.name_scope(self.scope):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor

            if self.masking:
                nelem = _nelem(y_true)
                y_true = _nan2zero(y_true)

            # Clip theta
            theta = tf.minimum(self.theta, 1e6)

            t1 = tf.math.lgamma(theta+eps) + tf.math.lgamma(y_true+1.0) - tf.math.lgamma(y_true+theta+eps)
            t2 = (theta+y_true) * tf.math.log(1.0 + (y_pred/(theta+eps))) + (y_true * (tf.math.log(theta+eps) - tf.math.log(y_pred+eps)))

            if self.debug:
                assert_ops = [
                    tf.debugging.assert_all_finite(y_pred, 'y_pred has inf/nans'),
                    tf.debugging.assert_all_finite(t1, 't1 has inf/nans'),
                    tf.debugging.assert_all_finite(t2, 't2 has inf/nans')]

                tf.summary.histogram('t1', t1)
                tf.summary.histogram('t2', t2)

                with tf.control_dependencies(assert_ops):
                    final = t1 + t2

            else:
                final = t1 + t2

            final = _nan2inf(final)

            if mean:
                if self.masking:
                    final = tf.divide(tf.reduce_sum(final), nelem)
                else:
                    final = tf.reduce_mean(final)


        return final
# ===================================================================
# Models
# ===================================================================
class Autoencoder(object):
    def __init__(self,
                 input_size,
                 output_size=None,
                 hidden_size=(64, 32, 64),
                 l2_coef=0.,
                 l1_coef=0.,
                 l2_enc_coef=0.,
                 l1_enc_coef=0.,
                 ridge=0.,
                 hidden_dropout=0.,
                 input_dropout=0.,
                 batchnorm=True,
                 activation='relu',
                 init='glorot_uniform',
                 file_path=None,
                 debug=False):

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.l2_coef = l2_coef
        self.l1_coef = l1_coef
        self.l2_enc_coef = l2_enc_coef
        self.l1_enc_coef = l1_enc_coef
        self.ridge = ridge
        self.hidden_dropout = hidden_dropout
        self.input_dropout = input_dropout
        self.batchnorm = batchnorm
        self.activation = activation
        self.init = init
        self.loss = None
        self.file_path = file_path
        self.extra_models = {}
        self.model = None
        self.encoder = None
        self.decoder = None
        self.input_layer = None
        self.sf_layer = None
        self.debug = debug

        if self.output_size is None:
            self.output_size = input_size

        if isinstance(self.hidden_dropout, list):
            assert len(self.hidden_dropout) == len(self.hidden_size)
        else:
            self.hidden_dropout = [self.hidden_dropout]*len(self.hidden_size)

    def build(self):

        self.input_layer = tfk.layers.Input(shape=(self.input_size,), name='count')
        self.sf_layer = tfk.layers.Input(shape=(1,), name='size_factors')
        last_hidden = self.input_layer

        if self.input_dropout > 0.0:
            last_hidden = tfk.layers.Dropout(self.input_dropout, name='input_dropout')(last_hidden)

        for i, (hid_size, hid_drop) in enumerate(zip(self.hidden_size, self.hidden_dropout)):
            center_idx = int(np.floor(len(self.hidden_size) / 2.0))
            if i == center_idx:
                layer_name = 'center'
                stage = 'center'  # let downstream know where we are
            elif i < center_idx:
                layer_name = 'enc%s' % i
                stage = 'encoder'
            else:
                layer_name = 'dec%s' % (i-center_idx)
                stage = 'decoder'

            # use encoder-specific l1/l2 reg coefs if given
            if self.l1_enc_coef != 0. and stage in ('center', 'encoder'):
                l1 = self.l1_enc_coef
            else:
                l1 = self.l1_coef

            if self.l2_enc_coef != 0. and stage in ('center', 'encoder'):
                l2 = self.l2_enc_coef
            else:
                l2 = self.l2_coef

            last_hidden = tfk.layers.Dense(hid_size, activation=None, kernel_initializer=self.init,
                                kernel_regularizer = tfk.regularizers.l1_l2(l1, l2),
                                name=layer_name)(last_hidden)
            if self.batchnorm:
                last_hidden = tfk.layers.BatchNormalization(center=True, scale=False)(last_hidden)

            # Use separate act. layers to give user the option to get pre-activations
            # of layers when requested
            if self.activation in advanced_activations:
                last_hidden = tfk.layers.__dict__[self.activation](name='%s_act'%layer_name)(last_hidden)
            else:
                last_hidden = tfk.layers.Activation(self.activation, name='%s_act'%layer_name)(last_hidden)

            if hid_drop > 0.0:
                last_hidden = tfk.layers.Dropout(hid_drop, name='%s_drop'%layer_name)(last_hidden)

        self.decoder_output = last_hidden
        self.build_output()

    def build_output(self):

        self.loss = tfk.losses.mean_squared_error
        mean = tfk.layers.Dense(self.output_size, kernel_initializer=self.init,
                     kernel_regularizer = tfk.regularizers.l1_l2(self.l1_coef, self.l2_coef),
                     name='mean')(self.decoder_output)
        output = ColwiseMultLayer([mean, self.sf_layer])

        # keep unscaled output as an extra model
        self.extra_models['mean_norm'] = tfk.Model(inputs=self.input_layer, outputs=mean)
        self.extra_models['decoded'] = tfk.Model(inputs=self.input_layer, outputs=self.decoder_output)
        self.model = tfk.Model(inputs=[self.input_layer, self.sf_layer], outputs=output)

        self.encoder = self.get_encoder()

    def save(self):
        if self.file_path:
            os.makedirs(self.file_path, exist_ok=True)
            with open(os.path.join(self.file_path, 'model.pickle'), 'wb') as f:
                pickle.dump(self, f)

    def load_weights(self, filename):
        self.model.load_weights(filename)
        self.encoder = self.get_encoder()
        self.decoder = None  # get_decoder()

    def get_decoder(self):
        i = 0
        for l in self.model.layers:
            if l.name == 'center_drop':
                break
            i += 1

        return tfk.Model(inputs=self.model.get_layer(index=i+1).input,
                     outputs=self.model.output)

    def get_encoder(self, activation=False):
        if activation:
            ret = tfk.Model(inputs=self.model.input,
                        outputs=self.model.get_layer('center_act').output)
        else:
            ret = tfk.Model(inputs=self.model.input,
                        outputs=self.model.get_layer('center').output)
        return ret

    def predict(self, adata, mode='denoise', return_info=False, copy=False):

        assert mode in ('denoise', 'latent', 'full'), 'Unknown mode'

        adata = adata.copy() if copy else adata

        if mode in ('denoise', 'full'):
            print('dca: Calculating reconstructions...')

            adata.X = self.model.predict({'count': adata.X,
                                          'size_factors': adata.obs.size_factors})

            #adata.uns['dca_loss'] = self.model.test_on_batch({'count': adata.X,
            #                                                  'size_factors': adata.obs.size_factors},
            #                                                 adata.raw.X)
        if mode in ('latent', 'full'):
            print('dca: Calculating low dimensional representations...')

            adata.obsm['X_dca'] = self.encoder.predict({'count': adata.X,
                                                        'size_factors': adata.obs.size_factors})
        if mode == 'latent':
            adata.X = adata.raw.X.copy() #recover normalized expression values

        return adata if copy else None

    # def write(self, adata, file_path, mode='denoise', colnames=None):
    #
    #     colnames = adata.var_names.values if colnames is None else colnames
    #     rownames = adata.obs_names.values
    #
    #     print('dca: Saving output(s)...')
    #     os.makedirs(file_path, exist_ok=True)
    #
    #     if mode in ('denoise', 'full'):
    #         print('dca: Saving denoised expression...')
    #         write_text_matrix(adata.X,
    #                           os.path.join(file_path, 'mean.tsv'),
    #                           rownames=rownames, colnames=colnames, transpose=True)
    #
    #     if mode in ('latent', 'full'):
    #         print('dca: Saving latent representations...')
    #         write_text_matrix(adata.obsm['X_dca'],
    #                           os.path.join(file_path, 'latent.tsv'),
    #                           rownames=rownames, transpose=False)


class NBAutoencoder(Autoencoder):

    def build_output(self):
        disp = tfk.layers.Dense(self.output_size, activation = DispAct,
                     kernel_initializer = self.init,
                     kernel_regularizer = tfk.regularizers.l1_l2(self.l1_coef, self.l2_coef),
                     name='dispersion')(self.decoder_output)

        mean = tfk.layers.Dense(self.output_size, activation = MeanAct, kernel_initializer=self.init,
                     kernel_regularizer = tfk.regularizers.l1_l2(self.l1_coef, self.l2_coef),
                     name='mean')(self.decoder_output)
        output = ColwiseMultLayer([mean, self.sf_layer])
        output = SliceLayer(0, name='slice')([output, disp])

        nb = NB(theta=disp, debug=self.debug)
        self.loss = nb.loss
        self.extra_models['dispersion'] = tfk.Model(inputs=self.input_layer, outputs=disp)
        self.extra_models['mean_norm'] = tfk.Model(inputs=self.input_layer, outputs=mean)
        self.extra_models['decoded'] = tfk.Model(inputs=self.input_layer, outputs=self.decoder_output)

        self.model = tfk.Model(inputs=[self.input_layer, self.sf_layer], outputs=output)

        self.encoder = self.get_encoder()

    def predict(self, adata, mode='denoise', return_info=False, copy=False):
        colnames = adata.var_names.values
        rownames = adata.obs_names.values

        res = super().predict(adata, mode, return_info, copy)
        adata = res if copy else adata

        if return_info:
            adata.obsm['X_dca_dispersion'] = self.extra_models['dispersion'].predict(adata.X)

        return adata if copy else None

    # def write(self, adata, file_path, mode='denoise', colnames=None):
    #     colnames = adata.var_names.values if colnames is None else colnames
    #     rownames = adata.obs_names.values
    #
    #     super().write(adata, file_path, mode, colnames=colnames)
    #
    #     if 'X_dca_dispersion' in adata.obsm_keys():
    #         write_text_matrix(adata.obsm['X_dca_dispersion'],
    #                           os.path.join(file_path, 'dispersion.tsv'),
    #                           colnames=colnames, transpose=True)

import random

def read_dataset(adata : Union[str, sc.AnnData], transpose : bool =False,
                 test_split: bool = False, copy : bool = False, check_counts : bool = True):

    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata, first_column_names=True)
    else:
        raise NotImplementedError

    if check_counts:
        # check if observations are unnormalized using first 10
        X_subset = adata.X[:10]
        norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
        if sparse.issparse(X_subset):
            assert (X_subset.astype(int) != X_subset).nnz == 0, norm_error
        else:
            assert np.all(X_subset.astype(int) == X_subset), norm_error

    if transpose: adata = adata.transpose()

    if test_split:
        train_idx, test_idx = skms.train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['dca_split'] = spl.values
    else:
        adata.obs['dca_split'] = 'train'

    adata.obs['dca_split'] = adata.obs['dca_split'].astype('category')
    print('dca: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    return adata


class DCAHandler(object):
    def __init__(self,
            # mode = 'denoise',
            # ae_type = 'nb-conddisp',
            hidden_size = (64, 32, 64), # network args
            hidden_dropout = 0.,
            batchnorm = True,
            activation = 'relu',
            init = 'glorot_uniform',
            network_kwds = {},
            epochs = 300,               # training args
            reduce_lr = 10,
            early_stop = 15,
            batch_size = 32,
            optimizer = 'Adam',
            learning_rate = None,
            random_state = 0,
            threads = None,
            verbose = False,
            training_kwds = {},
            copy = False,
            check_counts = True,
            ):
        """Deep count autoencoder(DCA) API.

        Fits a count autoencoder to the count data given in the anndata object
        in order to denoise the data and capture hidden representation of
        cells in low dimensions. Type of the autoencoder and return values are
        determined by the parameters.

        Parameters
        ----------
        adata : :class:`~scanpy.api.AnnData`
            An anndata file with `.raw` attribute representing raw counts.
        mode : `str`, optional. `denoise`(default), or `latent`.
            `denoise` overwrites `adata.X` with denoised expression values.
            In `latent` mode DCA adds `adata.obsm['X_dca']` to given adata
            object. This matrix represent latent representation of cells via DCA.
        ae_type : `str`, optional. `nb-conddisp`(default), `zinb`, `nb-conddisp` or `nb`.
            Type of the autoencoder. Return values and the architecture is
            determined by the type e.g. `nb` does not provide dropout
            probabilities.
        hidden_size : `tuple` or `list`, optional. Default: (64, 32, 64).
            Width of hidden layers.
        hidden_dropout : `float`, `tuple` or `list`, optional. Default: 0.0.
            Probability of weight dropout in the autoencoder (per layer if list
            or tuple).
        batchnorm : `bool`, optional. Default: `True`.
            If true, batch normalization is performed.
        activation : `str`, optional. Default: `relu`.
            Activation function of hidden layers.
        init : `str`, optional. Default: `glorot_uniform`.
            Initialization method used to initialize weights.
        network_kwds : `dict`, optional.
            Additional keyword arguments for the autoencoder.
        epochs : `int`, optional. Default: 300.
            Number of total epochs in training.
        reduce_lr : `int`, optional. Default: 10.
            Reduces learning rate if validation loss does not improve in given number of epochs.
        early_stop : `int`, optional. Default: 15.
            Stops training if validation loss does not improve in given number of epochs.
        batch_size : `int`, optional. Default: 32.
            Number of samples in the batch used for SGD.
        learning_rate : `float`, optional. Default: None.
            Learning rate to use in the training.
        optimizer : `str`, optional. Default: "RMSprop".
            Type of optimization method used for training.
        random_state : `int`, optional. Default: 0.
            Seed for python, numpy and tensorflow.
        threads : `int` or None, optional. Default: None
            Number of threads to use in training. All cores are used by default.
        verbose : `bool`, optional. Default: `False`.
            If true, prints additional information about training and architecture.
        training_kwds : `dict`, optional.
            Additional keyword arguments for the training process.
        return_model : `bool`, optional. Default: `False`.
            If true, trained autoencoder object is returned. See "Returns".
        return_info : `bool`, optional. Default: `False`.
            If true, all additional parameters of DCA are stored in `adata.obsm` such as dropout
            probabilities (obsm['X_dca_dropout']) and estimated dispersion values
            (obsm['X_dca_dispersion']), in case that autoencoder is of type
            zinb or zinb-conddisp.
        copy : `bool`, optional. Default: `False`.
            If true, a copy of anndata is returned.
        check_counts : `bool`. Default `True`.
            Check if the counts are unnormalized (raw) counts.

        Returns
        -------
        If `copy` is true and `return_model` is false, AnnData object is returned.

        In "denoise" mode, `adata.X` is overwritten with the denoised values. In "latent" mode, latent
        low dimensional representation of cells are stored in `adata.obsm['X_dca']` and `adata.X`
        is not modified. Note that these values are not corrected for library size effects.

        If `return_info` is true, all estimated distribution parameters are stored in AnnData such as:

        - `.obsm["X_dca_dropout"]` which is the mixture coefficient (pi) of the zero component
        in ZINB, i.e. dropout probability. (Only if ae_type is zinb or zinb-conddisp)

        - `.obsm["X_dca_dispersion"]` which is the dispersion parameter of NB.

        - `.uns["dca_loss_history"]` which stores the loss history of the training.

        Finally, the raw counts are stored as `.raw`.

        If `return_model` is given, trained model is returned. When both `copy` and `return_model`
        are true, a tuple of anndata and model is returned in that order.
        """

        #assert mode in ('denoise', 'latent'), '%s is not a valid mode.' % mode


        # self.mode = 'denoise',
        # self.ae_type = 'nb-conddisp',
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.batchnorm = batchnorm
        self.activation = activation
        self.init = init
        self.network_kwds = network_kwds
        self.epochs = epochs
        self.reduce_lr = reduce_lr
        self.early_stop = early_stop
        self.batch_size = batch_size
        self.optimizer = tfk.optimizers.__dict__[optimizer] if (isinstance(optimizer, str)) else optimizer
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.threads = threads
        self.verbose = verbose
        self.training_kwds = training_kwds
        self.copy = copy
        self.check_counts = check_counts

        self.adata = None
        self.network = None


    def build(self, adata : sc.AnnData, normalize_per_cell : bool = True,
              scale : bool = True, log1p : bool = True) -> None:
        '''
        :param adata
            Annotated data
        :param normalize_per_cell : `bool`, optional. Default: `True`.
            If true, library size normalization is performed using
            the `sc.pp.normalize_per_cell` function in Scanpy and saved into adata
            object. Mean layer is re-introduces library size differences by
            scaling the mean value of each cell in the output layer. See the
            manuscript for more details.
        :param scale : `bool`, optional. Default: `True`.
            If true, the input of the autoencoder is centered using
            `sc.pp.scale` function of Scanpy. Note that the output is kept as raw
            counts as loss functions are designed for the count data.
        log1p : `bool`, optional. Default: `True`.
            If true, the input of the autoencoder is log transformed with a
            pseudocount of one using `sc.pp.log1p` function of Scanpy.


        :return:
        '''

        #hist = train(adata[adata.obs.dca_split == 'train'], net, **training_kwds)
        # res = net.predict(adata, mode, return_info, copy)
        # adata = res if copy else adata
        #
        # if return_info:
        #     adata.uns['dca_loss_history'] = hist.history
        #
        # if return_model:
        #     return (adata, net) if copy else net
        # else:
        #     return adata if copy else None
        # this creates adata.raw with raw counts and copies adata if copy==True
        assert isinstance(adata, sc.AnnData), 'adata must be an AnnData instance'
        adata = read_dataset(adata,
                             transpose = False,
                             test_split = False,
                             copy = self.copy,
                             check_counts = self.check_counts)

        # check for zero genes
        nonzero_genes, _ = sc.pp.filter_genes(adata.X, min_counts=1)
        assert nonzero_genes.all(), 'Please remove all-zero genes before using DCA.'

        adata = normalize(adata,
                          filter_min_counts=False, # no filtering, keep cell and gene idxs same
                          size_factors=normalize_per_cell,
                          normalize_input=scale,
                          logtrans_input=log1p)

        self.adata = adata

        # set seed for reproducibility
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        os.environ['PYTHONHASHSEED'] = '0'

        network_kwds = {**self.network_kwds,
                        'hidden_size': self.hidden_size,
                        'hidden_dropout': self.hidden_dropout,
                        'batchnorm': self.batchnorm,
                        'activation': self.activation,
                        'init': self.init
                        }

        # TODO Is this necessary?
        # from tensorflow.python.framework.ops import disable_eager_execution
        # disable_eager_execution()

        input_size = output_size = adata.n_vars
        self.network = NBAutoencoder(input_size = input_size, output_size = output_size, **network_kwds)
        # net.save()
        self.network.build()

        training_kwds = {**self.training_kwds,
                         'epochs': self.epochs,
                         'reduce_lr': self.reduce_lr,
                         'early_stop': self.early_stop,
                         'batch_size': self.batch_size,
                         'optimizer': self.optimizer,
                         'verbose': self.verbose,
                         'threads': self.threads,
                         'learning_rate': self.learning_rate
                         }
        self.network.model.summary()

    def train(self, use_raw_as_output : bool = True, validation_split :float = 0.1, clip_grad :float = 5.0):
        model = self.network.model
        loss = self.network.loss
        adata = self.adata
        print("Data type: ", type(adata.X))

        if self.learning_rate is None:
            optimizer = self.optimizer(clipvalue = clip_grad)
        else:
            optimizer = self.optimizer(lr = self.learning_rate, clipvalue = clip_grad)

        model.compile(loss = loss, optimizer = optimizer, run_eagerly = True)

        inputs = {'count': adata.X, 'size_factors': adata.obs['size_factors']}

        output = adata.raw.X if use_raw_as_output else adata.X

        # TODO Add callbacks
        callbacks = []

        log.info("Training started")

        hist = model.fit(inputs, output,
                         epochs = self.epochs,
                         batch_size = self.batch_size,
                         shuffle = True,
                         callbacks = callbacks,
                         validation_split = validation_split,
                         verbose  = self.verbose,
                         **self.training_kwds)

        log.info("Training completed")

        return hist