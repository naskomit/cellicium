import numpy as np
import pandas as pd
import scanpy as sc
import tensorflow as tf
import tensorflow.keras as tfk
import scipy.sparse as sparse
import numba as nmb
import typing as tp
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets
import cellicium.scrna as crna
from cellicium.logging import logger as log

class DoubleTripletDistanceLayer(tfk.layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """
    def __init__(self, correspondance_coeff, margin, **kwargs):
        super().__init__(**kwargs)
        self.correspondance_coeff = correspondance_coeff
        self.margin = margin

    @staticmethod
    def distance_2(x1, x2):
        return tf.reduce_sum(tf.square(x1 - x2), axis = -1)

    def call(self, input, *args, **kwargs):
        ((anchor_z1, positive_z1, negative_z1), (anchor_z2, positive_z2, negative_z2)) = input
        loss_corr = self.distance_2(anchor_z1, anchor_z2) + self.distance_2(positive_z1, positive_z2) + self.distance_2(negative_z1, negative_z2)
        loss_corr *= self.correspondance_coeff

        ap_distance = self.distance_2(anchor_z1, positive_z1)
        an_distance = self.distance_2(anchor_z1, negative_z1)
        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss_pn = ap_distance - an_distance
        loss_pn = tf.maximum(loss_pn + self.margin, 0.0)

        return loss_corr + loss_pn

class SiamesseModel(tfk.Model):
    def __init__(self, dim_in1, dim_in2, dim_z = 20, margin = 1.0, correspondance_coeff = 0.01, verbose = False):
        super().__init__()
        self.dim_in1 = dim_in1
        self.dim_in2 = dim_in2
        self.dim_z = dim_z
        self.margin = margin
        self.correspondance_coeff = correspondance_coeff
        self.model1, self.model2, self.network = self.build_models()
        self.loss_tracker = tfk.metrics.Mean(name = "loss")
        self.verbose = verbose
        if self.verbose:
            self.network.summary()

    def build_models(self) -> tp.Tuple[tfk.Model, tfk.Model, tfk.Model]:
        dropout_rate = 0.2
        input1 = tfk.Input(self.dim_in1, name = 'mod1 input')
        x1 = input1
        # x1 = tfk.layers.Dense(self.dim_z, activation = 'ReLU')(x1)
        # x1 = tfk.layers.Dropout(dropout_rate)(x1)
        x1 = tfk.layers.Dense(self.dim_z)(x1)
        z1 = tfk.layers.Dropout(dropout_rate)(x1)
        model1 = tfk.Model(input1, z1)

        input2 = tfk.Input(self.dim_in2, name = 'mod2 input')
        x2 = input2
        # x2 = tfk.layers.Dense(self.dim_z, activation = 'ReLU')(x2)
        # x2 = tfk.layers.Dropout(dropout_rate)(x2)
        x2 = tfk.layers.Dense(self.dim_z)(x2)
        z2 = tfk.layers.Dropout(dropout_rate)(x2)
        model2 = tfk.Model(input2, z2)

        anchor_index = tfk.Input(name = 'anchor_index', shape = 1)
        anchor_input_mod1 = tfk.Input(name = 'anchor_mod1', shape = self.dim_in1)
        positive_input_mod1 = tfk.Input(name = 'positive_mod1', shape = self.dim_in1)
        negative_input_mod1 = tfk.Input(name = 'negative_mod1', shape = self.dim_in1)
        anchor_input_mod2 = tfk.Input(name = 'anchor_mod2', shape = self.dim_in2)
        positive_input_mod2 = tfk.Input(name = 'positive_mod2', shape = self.dim_in2)
        negative_input_mod2 = tfk.Input(name = 'negative_mod2', shape = self.dim_in2)

        anchor_z1 = model1(anchor_input_mod1)
        positive_z1 = model1(positive_input_mod1)
        negative_z1 = model1(negative_input_mod1)

        anchor_z2 = model2(anchor_input_mod2)
        positive_z2 = model2(positive_input_mod2)
        negative_z2 = model2(negative_input_mod2)

        distances = DoubleTripletDistanceLayer(self.correspondance_coeff, self.margin)(
            ((anchor_z1, positive_z1, negative_z1), (anchor_z2, positive_z2, negative_z2))
        )

        network = tfk.Model(inputs = [anchor_index, anchor_input_mod1, positive_input_mod1, negative_input_mod1, anchor_input_mod2, positive_input_mod2, negative_input_mod2],
                            outputs = distances, name = 'SiameseModelNetwork')

        return model1, model2, network

    def call(self, inputs, *args, **kwargs):
        return self.network(inputs)


    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)
        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is the loss
        loss = self.network(data)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]

class EpochProgressCallback(tfk.callbacks.Callback):
    def __init__(self, total_num_epochs : int):
        from tqdm.notebook import tqdm
        super().__init__()
        self.total_num_epochs = total_num_epochs
        # Outputs
        progress_bar_out = widgets.Output(layout = {'width': '600px', 'height': '50px'})
        self.metrics_out = widgets.Output(layout = {'width': '600px', 'height': '100px'})
        self.fig_out = widgets.Output(layout = {'margin': '50px 0px 50px 50px'})

        # self.stop_button = widgets.Button(description = "Stop training")
        # def on_stop_button(instance):
        #     with self.metrics_out:
        #         print('Stopping ....')
        #         self.model.stop_training = True
        #         raise ValueError("Stopped...")
        # self.stop_button.on_click(on_stop_button)

        training_output = widgets.HBox([widgets.VBox([progress_bar_out, self.metrics_out]), self.fig_out], layout={'border': '1px solid black', 'height': '300px'})
        display(training_output)
        with progress_bar_out:
            self.pbar = tqdm(total = total_num_epochs)
        # self.fig = plt.figure()
        self.metrics = {}
        self.first_run_completed = False

    def plot_metrics(self):
        fig = plt.gcf()
        ax = fig.gca()
        ax.clear()
        for k, v in self.metrics.items():
            ax.plot(v, label = k)
        ax.legend()
        ax.set_yscale('log')
        self.fig_out.clear_output()
        with self.fig_out:
            display(fig)

    def on_epoch_end(self, epoch, logs = None):
        self.pbar.update(1)

        if not self.first_run_completed:
            for k, v, in logs.items():
                self.metrics[k] = []

        for k, v, in logs.items():
            self.metrics[k].append(v)
            self.metrics_out.clear_output()
            with self.metrics_out:
                print(logs)

        if self.first_run_completed and (epoch % 20 == 0):
            self.plot_metrics()

        self.first_run_completed = True

    def on_train_end(self, logs=None):
        self.pbar.close()
        self.plot_metrics()
        plt.close()

class SiameseModelManager:
    def __init__(self, dim_z : int = 20, margin : float = 1.0, correspondance_coeff : float = 0.01, seed : int = 1234, verbose : bool = False):
        self.dim_z = dim_z
        self.margin = margin
        self.correspondance_coeff = correspondance_coeff
        self.seed = seed
        self.verbose = verbose
        self.model = None
        self.modalities = {}
        self.progress_tracker = None
        self.train_dataset = None
        self.val_dataset = None

    def train(self, adata1 : sc.AnnData, adata2 : sc.AnnData, dataset_gen, gen_kwargs = None, lr = 0.0001, epochs = 100):
        if self.model is None:
            self.model = SiamesseModel(
                dim_in1 = adata1.n_vars, dim_in2 = adata2.n_vars, dim_z = self.dim_z,
                margin = self.margin, correspondance_coeff = self.correspondance_coeff, verbose = self.verbose)
            self.modality_encoders = {adata1.uns['modality'] : self.model.model1, adata2.uns['modality'] : self.model.model2}
        tf.random.set_seed(self.seed)

        if gen_kwargs == None:
            gen_kwargs = {}
        dataset, n_data_points = dataset_gen(adata1, adata2, **gen_kwargs)
        dataset = dataset.shuffle(buffer_size = 8192)

        self.train_dataset = dataset.take(round(n_data_points * 0.8))
        self.val_dataset = dataset.skip(round(n_data_points * 0.8))

        self.train_dataset = self.train_dataset.batch(256, drop_remainder=False)
        self.val_dataset = self.val_dataset.batch(256, drop_remainder=False)

        self.progress_tracker = EpochProgressCallback(epochs)
        self.model.compile(optimizer = tfk.optimizers.Adam(lr)) # , run_eagerly = True
        training_log = self.model.fit(
            self.train_dataset, epochs = epochs, validation_data = self.val_dataset,
            verbose = False, callbacks = [self.progress_tracker]
        )
        self.history = training_log.history

    def extend_train(self, lr = 0.0001, epochs = 100):
        if (self.train_dataset is None) or (self.val_dataset is None):
            raise RuntimeError('Cannot extend training if no initial training is performed')
        self.model.compile(optimizer = tfk.optimizers.Adam(lr))
        training_log = self.model.fit(
            self.train_dataset, epochs = epochs, validation_data = self.val_dataset,
            verbose = False, callbacks = [self.progress_tracker]
        )
        for k,v in training_log.history.items():
            self.history[k] += v

    def latent_representation(self, adata_list : tp.List[sc.AnnData]):
        Z_list = []
        obs_list = []
        for adata in adata_list:
            X = adata.X.todense() if sparse.issparse(adata.X) else adata.X
            modality = adata.uns['modality']
            Z_predict = self.modality_encoders[modality].predict(X)
            Z_list.append(Z_predict)
            obs = adata.obs[['cell_type', 'batch', 'site', 'donor']].copy()
            obs['modality'] = modality
            obs.index = obs.index + ('_' + modality)
            obs_list.append(obs)

        result = sc.AnnData(X = np.vstack(Z_list), obs = pd.concat(obs_list))
        return result

    def plot_integration_umap(self, adata_list : tp.List[sc.AnnData]):
        adata = self.latent_representation(adata_list)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        fg = crna.pl.figure_grid(ncol = 2, nrow = 1, figsize = (40, 20))
        sc.pl.umap(adata, color = 'modality', ax = next(fg), s = 20, show = False)
        sc.pl.umap(adata, color = 'cell_type', ax = next(fg), s = 20, show = False, legend_loc = 'on data')
        return adata

    def compute_match(self, adata1 : sc.AnnData, adata2 : sc.AnnData):
        # val_ind = np.array([i[0].numpy() for i in self.val_dataset.unbatch()])
        # adata1 = adata1[val_ind, ]
        z1 = self.latent_representation([adata1])
        z2 = self.latent_representation([adata2])


    def create_triplets_dataset_by_group(self, adata1 : sc.AnnData, adata2 : sc.AnnData, group_column):
        n_obs = adata1.n_obs
        groups = adata1.obs[[group_column]].reset_index().groupby(group_column).groups
        anchor_items_mod1 = np.zeros((n_obs, adata1.n_vars), dtype = 'float32')
        positive_items_mod1 = np.zeros((n_obs, adata1.n_vars), dtype = 'float32')
        negative_items_mod1 = np.zeros((n_obs, adata1.n_vars), dtype = 'float32')
        anchor_items_mod2 = np.zeros((n_obs, adata2.n_vars), dtype = 'float32')
        positive_items_mod2 = np.zeros((n_obs, adata2.n_vars), dtype = 'float32')
        negative_items_mod2 = np.zeros((n_obs, adata2.n_vars), dtype = 'float32')

        @nmb.jit(nopython=True)
        def optimized_create(labels, groups, X1, X2, outputs, seed):
            [anchor_items_mod1, positive_items_mod1, negative_items_mod1,
             anchor_items_mod2, positive_items_mod2, negative_items_mod2] = outputs
            np.random.seed(seed)

            for i in range(n_obs):
                group_id = labels[i]
                anchor_items_mod1[i, :] = X1[i, :]
                anchor_items_mod2[i, :] = X2[i, :]
                similar_items = groups[group_id]
                pos_example_id = np.random.choice(similar_items)
                positive_items_mod1[i, :] = X1[pos_example_id, :]
                positive_items_mod2[i, :] = X2[pos_example_id, :]

                selected_negative_example = False
                while not selected_negative_example:
                    negative_example_id = np.random.choice(n_obs)
                    if labels[negative_example_id] != group_id:
                        negative_items_mod1[i, :] = X1[negative_example_id, :]
                        negative_items_mod2[i, :] = X2[negative_example_id, :]
                        selected_negative_example = True

        X1 = adata1.X.todense() if sparse.issparse(adata1.X) else adata1.X
        X2 = adata2.X.todense() if sparse.issparse(adata2.X) else adata2.X
        groups_typed = nmb.typed.Dict.empty(
            key_type = nmb.core.types.unicode_type,
            value_type = nmb.core.types.int64[:]
        )
        for k, v in groups.items():
            groups_typed[k] = groups[k].values
        labels = list(adata1.obs[group_column].values)

        outputs = [anchor_items_mod1, positive_items_mod1, negative_items_mod1,
                   anchor_items_mod2, positive_items_mod2, negative_items_mod2]
        optimized_create(labels = labels, groups = groups_typed, X1 = X1, X2 = X2,
                         outputs = outputs, seed = self.seed)
        outputs = [np.arange(n_obs)] + outputs
        for i in range(len(outputs)):
            outputs[i] = tf.data.Dataset.from_tensor_slices(outputs[i])

        outputs = tf.data.Dataset.zip(tuple(outputs))

        return outputs, anchor_items_mod1.shape[0]

    def create_triplets_dataset_by_neighbours(self, adata1 : sc.AnnData, adata2 : sc.AnnData, **kwargs):
        n_obs = adata1.n_obs
        anchor_items_mod1 = np.zeros((n_obs, adata1.n_vars), dtype = 'float32')
        positive_items_mod1 = np.zeros((n_obs, adata1.n_vars), dtype = 'float32')
        negative_items_mod1 = np.zeros((n_obs, adata1.n_vars), dtype = 'float32')
        anchor_items_mod2 = np.zeros((n_obs, adata2.n_vars), dtype = 'float32')
        positive_items_mod2 = np.zeros((n_obs, adata2.n_vars), dtype = 'float32')
        negative_items_mod2 = np.zeros((n_obs, adata2.n_vars), dtype = 'float32')
        log.info('Computing neighbours...')
        sc.pp.neighbors(adata1, **kwargs)
        log.info('Done computing neighbours!')

        X1 = adata1.X.todense() if sparse.issparse(adata1.X) else adata1.X
        X2 = adata2.X.todense() if sparse.issparse(adata2.X) else adata2.X
        neighbours = adata1.obsp['connectivities']
        rng = np.random.default_rng(seed = self.seed)

        log.info('Creating triplets ...')
        for i in range(n_obs):
            anchor_items_mod1[i, :] = X1[i, :]
            anchor_items_mod2[i, :] = X2[i, :]

            _, neighb_ind = neighbours[i, :].nonzero()
            pos_example_id = rng.choice(neighb_ind)
            positive_items_mod1[i, :] = X1[pos_example_id, :]
            positive_items_mod2[i, :] = X2[pos_example_id, :]

            selected_negative_example = False
            while not selected_negative_example:
                negative_example_id = rng.choice(n_obs)
                if not(np.isin(negative_example_id, neighb_ind)):
                    negative_items_mod1[i, :] = X1[negative_example_id, :]
                    negative_items_mod2[i, :] = X2[negative_example_id, :]
                    selected_negative_example = True
        log.info('Done creating triplets!')

        outputs = [np.arange(n_obs), anchor_items_mod1, positive_items_mod1, negative_items_mod1,
                   anchor_items_mod2, positive_items_mod2, negative_items_mod2]

        for i in range(len(outputs)):
            outputs[i] = tf.data.Dataset.from_tensor_slices(outputs[i])

        outputs = tf.data.Dataset.zip(tuple(outputs))

        return outputs, anchor_items_mod1.shape[0]




