import numpy as np
import pandas as pd
import scanpy as sc
import tensorflow as tf
import tensorflow.keras as tfk
import scipy.sparse as sparse
import numba as nmb
import typing as tp
import matplotlib.pyplot as plt
import os
import cellicium.scrna as crna
from cellicium.logging import logger as log
from .nn_utils import EpochProgressCallback, LinLogLayer
from . import base

class DoubleTripletDistanceLayer(tfk.layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """
    def __init__(self, margin, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    @staticmethod
    def distance_2(x1, x2):
        return tf.reduce_sum(tf.square(x1 - x2), axis = -1)

    def call(self, input, *args, **kwargs):
        ((anchor_z1, positive_z1, negative_z1), (anchor_z2, positive_z2, negative_z2)) = input
        loss_corresp = (self.distance_2(anchor_z1, anchor_z2) +
                        self.distance_2(positive_z1, positive_z2) +
                        self.distance_2(negative_z1, negative_z2))

        ap_distance = self.distance_2(anchor_z1, positive_z1)
        an_distance = self.distance_2(anchor_z1, negative_z1)
        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss_pos_neg = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
        loss = {'corresp_loss': tf.reduce_mean(loss_corresp), 'pos_neg_loss': tf.reduce_mean(loss_pos_neg)}
        return loss


class SiamesseModel(base.EncoderModel):
    def __init__(self, dim_in1, dim_in2, dim_z = 20, margin = 1.0, correspondance_coeff = 0.01):
        super().__init__()
        self.dim_in1 = dim_in1
        self.dim_in2 = dim_in2
        self.dim_z = dim_z
        self.margin = margin
        self.correspondance_coeff = correspondance_coeff
        self.add_loss_tracker("corresp_loss", gain = correspondance_coeff)
        self.add_loss_tracker("pos_neg_loss")
        self.network = self.build_model()

    def assemble_encoder(self, input, dropout_rate = 0.3, log_offset = 1.0):
        x = input
        # x = tfk.layers.Dense(self.dim_z)(x)
        x = LinLogLayer(shape = self.dim_z, log_offset = log_offset)(x)
        x = tfk.layers.BatchNormalization()(x)
        x = tfk.layers.Dropout(dropout_rate)(x)
        z = x
        model = tfk.Model(input, z)
        return model

    def build_model(self) -> tfk.Model:
        input1 = tfk.Input(self.dim_in1, name = 'mod1 input')
        model1 = self.assemble_encoder(input1)

        input2 = tfk.Input(self.dim_in2, name = 'mod2 input')
        model2 = self.assemble_encoder(input2)

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

        # Add a layer computing the losses
        self.loss_layer = DoubleTripletDistanceLayer(self.margin)
        model_losses = self.loss_layer(((anchor_z1, positive_z1, negative_z1), (anchor_z2, positive_z2, negative_z2)))

        network = tfk.Model(inputs = [anchor_index, anchor_input_mod1, positive_input_mod1, negative_input_mod1, anchor_input_mod2, positive_input_mod2, negative_input_mod2],
                            outputs = model_losses, name = 'SiameseModelNetwork')
        self.model1 = model1
        self.model2 = model2
        return network


class SiameseModelManager(base.ModelManagerBase):
    def __init__(self, dim_z : int = 20, **kwargs):
        super().__init__(**kwargs)
        self.dim_z = dim_z
        self.modalities = {}
        self.modality_encoders = None

    def save_encoders(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
        for k,v in self.modality_encoders.items():
            v.save_weights(os.path.join(dir, f'{k}.h5'), save_format = 'hdf5')

    def load_encoders(self, dir, dims_dict):
        modalities = list(dims_dict.keys())
        if self.model is None:
            self.model = SiamesseModel(
                dim_in1 = dims_dict[modalities[0]], dim_in2 = dims_dict[modalities[1]], dim_z = self.dim_z)
        self.modality_encoders = {}
        for modality, model in zip(modalities, [self.model.model1, self.model.model2]):
            self.modality_encoders[modality] = model
            filepath = os.path.join(dir, modality + '.h5')
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
        margin = kwargs.pop('margin')
        correspondance_coeff = kwargs.pop('correspondance_coeff')
        model = SiamesseModel(
            dim_in1 = mod1_nvars, dim_in2 = mod2_nvars, dim_z = self.dim_z,
            margin = margin, correspondance_coeff = correspondance_coeff)

        self.modality_encoders = {
            modality1 : model.model1, modality2 : model.model2
        }

        return model

    def train(self, modalities, triplet_dataset, n_data_points, **kwargs):
        # triplet_dataset = triplet_dataset.shuffle(buffer_size = 8192)
        kwargs['mod1_nvars'] = modalities[0]['n_vars']
        kwargs['mod2_nvars'] = modalities[1]['n_vars']
        kwargs['modality1'] = modalities[0]['name']
        kwargs['modality2'] = modalities[1]['name']
        self.do_train(triplet_dataset, n_data_points, **kwargs)

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

    # def compute_match(self, adata1 : sc.AnnData, adata2 : sc.AnnData):
    #     # val_ind = np.array([i[0].numpy() for i in self.val_dataset.unbatch()])
    #     # adata1 = adata1[val_ind, ]
    #     z1 = self.latent_representation([adata1])
    #     z2 = self.latent_representation([adata2])



    # def create_triplets_dataset_by_group(self, adata1 : sc.AnnData, adata2 : sc.AnnData, group_column):
    #     n_obs = adata1.n_obs
    #     groups = adata1.obs[[group_column]].reset_index().groupby(group_column).groups
    #     anchor_items_mod1 = np.zeros((n_obs, adata1.n_vars), dtype = 'float32')
    #     positive_items_mod1 = np.zeros((n_obs, adata1.n_vars), dtype = 'float32')
    #     negative_items_mod1 = np.zeros((n_obs, adata1.n_vars), dtype = 'float32')
    #     anchor_items_mod2 = np.zeros((n_obs, adata2.n_vars), dtype = 'float32')
    #     positive_items_mod2 = np.zeros((n_obs, adata2.n_vars), dtype = 'float32')
    #     negative_items_mod2 = np.zeros((n_obs, adata2.n_vars), dtype = 'float32')
    #
    #     @nmb.jit(nopython=True)
    #     def optimized_create(labels, groups, X1, X2, outputs, seed):
    #         [anchor_items_mod1, positive_items_mod1, negative_items_mod1,
    #          anchor_items_mod2, positive_items_mod2, negative_items_mod2] = outputs
    #         np.random.seed(seed)
    #
    #         for i in range(n_obs):
    #             group_id = labels[i]
    #             anchor_items_mod1[i, :] = X1[i, :]
    #             anchor_items_mod2[i, :] = X2[i, :]
    #             similar_items = groups[group_id]
    #             pos_example_id = np.random.choice(similar_items)
    #             positive_items_mod1[i, :] = X1[pos_example_id, :]
    #             positive_items_mod2[i, :] = X2[pos_example_id, :]
    #
    #             selected_negative_example = False
    #             while not selected_negative_example:
    #                 negative_example_id = np.random.choice(n_obs)
    #                 if labels[negative_example_id] != group_id:
    #                     negative_items_mod1[i, :] = X1[negative_example_id, :]
    #                     negative_items_mod2[i, :] = X2[negative_example_id, :]
    #                     selected_negative_example = True
    #
    #     X1 = adata1.X.todense() if sparse.issparse(adata1.X) else adata1.X
    #     X2 = adata2.X.todense() if sparse.issparse(adata2.X) else adata2.X
    #     groups_typed = nmb.typed.Dict.empty(
    #         key_type = nmb.core.types.unicode_type,
    #         value_type = nmb.core.types.int64[:]
    #     )
    #     for k, v in groups.items():
    #         groups_typed[k] = groups[k].values
    #     labels = list(adata1.obs[group_column].values)
    #
    #     outputs = [anchor_items_mod1, positive_items_mod1, negative_items_mod1,
    #                anchor_items_mod2, positive_items_mod2, negative_items_mod2]
    #     optimized_create(labels = labels, groups = groups_typed, X1 = X1, X2 = X2,
    #                      outputs = outputs, seed = self.seed)
    #     outputs = [np.arange(n_obs)] + outputs
    #     for i in range(len(outputs)):
    #         outputs[i] = tf.data.Dataset.from_tensor_slices(outputs[i])
    #
    #     outputs = tf.data.Dataset.zip(tuple(outputs))
    #
    #     return outputs, anchor_items_mod1.shape[0]


def triplets_from_neighbours(adata1 : sc.AnnData, adata2 : sc.AnnData, **kwargs):
        seed = kwargs.pop('seed', 42)
        samples_per_anchor = kwargs.pop('samples_per_anchor', 1)
        assert (isinstance(samples_per_anchor, int) and samples_per_anchor >= 1)

        n_obs = adata1.n_obs
        n_out = n_obs * samples_per_anchor
        anchor_items_mod1 = np.zeros((n_out, adata1.n_vars), dtype = 'float32')
        positive_items_mod1 = np.zeros((n_out, adata1.n_vars), dtype = 'float32')
        negative_items_mod1 = np.zeros((n_out, adata1.n_vars), dtype = 'float32')
        anchor_items_mod2 = np.zeros((n_out, adata2.n_vars), dtype = 'float32')
        positive_items_mod2 = np.zeros((n_out, adata2.n_vars), dtype = 'float32')
        negative_items_mod2 = np.zeros((n_out, adata2.n_vars), dtype = 'float32')
        log.info('Computing neighbours...')
        sc.pp.neighbors(adata1, use_rep='X', **kwargs)
        log.info('Done computing neighbours!')

        X1 = adata1.X.todense() if sparse.issparse(adata1.X) else adata1.X
        X2 = adata2.X.todense() if sparse.issparse(adata2.X) else adata2.X
        neighbours = adata1.obsp['connectivities']
        rng = np.random.default_rng(seed = seed)

        log.info(f'Creating {n_out} triplets...')
        for i in range(n_obs):
            _, neighb_ind = neighbours[i, :].nonzero()
            pos_example_id = rng.choice(neighb_ind, size = samples_per_anchor, replace = False)
            for j in range(samples_per_anchor):
                ind = i * samples_per_anchor + j
                anchor_items_mod1[ind, :] = X1[i, :]
                anchor_items_mod2[ind, :] = X2[i, :]

                positive_items_mod1[ind, :] = X1[pos_example_id[j], :]
                positive_items_mod2[ind, :] = X2[pos_example_id[j], :]

                selected_negative_example = False
                while not selected_negative_example:
                    negative_example_id = rng.choice(n_obs)
                    if not(np.isin(negative_example_id, neighb_ind)):
                        negative_items_mod1[ind, :] = X1[negative_example_id, :]
                        negative_items_mod2[ind, :] = X2[negative_example_id, :]
                        selected_negative_example = True
        log.info('Done creating triplets!')

        outputs = [np.arange(n_out), anchor_items_mod1, positive_items_mod1, negative_items_mod1,
                   anchor_items_mod2, positive_items_mod2, negative_items_mod2]

        for i in range(len(outputs)):
            outputs[i] = tf.data.Dataset.from_tensor_slices(outputs[i])

        outputs = tf.data.Dataset.zip(tuple(outputs))

        return outputs, anchor_items_mod1.shape[0]




