import collections as coll
import scanpy as sc
import numpy as np
import abc
import typing as tp
import tensorflow as tf
import tensorflow.keras as tfk
from . import nn_utils as nnu

ProblemDatasetBase = coll.namedtuple(
    'ProblemDataset', [
        'train_mod1',
        'train_mod2',
        'train_sol',
        'test_mod1',
        'test_mod2',
        'test_sol',
        'modality1',
        'modality2'
    ]
)

class ProblemDataset(ProblemDatasetBase):
    __slots__ = ()
    def to_modality_dict(self, part = 'train'):
        if part == 'train':
            return {self.modality1: self.train_mod1, self.modality2: self.train_mod2}
        elif part == 'test':
            return {self.modality1: self.test_mod1, self.modality2: self.test_mod2}
        else:
            raise ValueError(f'part should be train or test, not {part}')

    @property
    def combined_mod1(self):
        result = sc.concat([self.train_mod1, self.test_mod1], axis = 0)
        result.uns['modality'] = self.modality1
        return result

    @property
    def combined_mod2(self):
        result = sc.concat([self.train_mod2, self.test_mod2], axis = 0)
        result.uns['modality'] = self.modality2
        return result

    def get_data(self, group, modality, sort = False):
        if modality == self.modality1:
            if group == 'train':
                if sort:
                    order = np.argsort(self.train_sol.X.nonzero()[1])
                    return self.train_mod1[order, :]
                else:
                    return self.train_mod1
            elif group == 'test':
                if sort:
                    order = np.argsort(self.test_sol.X.nonzero()[1])
                    return self.test_mod1[order, :]
                else:
                    return self.test_mod1
            else:
                raise ValueError(f'Group must be train or test, not {group}')
        elif modality == self.modality2:
            if group == 'train':
                return self.train_mod2
            elif group == 'test':
                return self.test_mod2
            else:
                raise ValueError(f'Group must be train or test, not {group}')
        else:
            raise ValueError(f'Modality must be {self.modality1} or {self.modality2}, not {modality}')

    def subset_samples(self, predicate):
        train_sel_mod1 = predicate(self.train_mod1, 'train', self.modality1)
        train_sel_mod2 = predicate(self.train_mod2, 'train', self.modality2)
        test_sel_mod1 = predicate(self.test_mod1, 'test', self.modality1)
        test_sel_mod2 = predicate(self.test_mod1, 'test', self.modality2)
        return ProblemDataset(
            train_mod1 = self.train_mod1[train_sel_mod1, :],
            train_mod2 = self.train_mod2[train_sel_mod2, :],
            train_sol = self.train_sol[train_sel_mod1, train_sel_mod2],
            test_mod1 = self.test_mod1[test_sel_mod1, :],
            test_mod2 = self.test_mod2[test_sel_mod2, :],
            test_sol= self.test_sol[test_sel_mod1, test_sel_mod2] if self.test_sol is not None else None,
            modality1 = self.modality1,
            modality2 = self.modality2
        )

class EncoderModel(tfk.Model):
    def __init__(self):
        super().__init__()
        self.training_plan = None
        self.loss_trackers = {}
        self.add_loss_tracker('reglr_loss')
        self.add_loss_tracker('total_loss')

    @abc.abstractmethod
    def build_model(self, **kwargs) -> tfk.Model:
        pass

    def add_loss_tracker(self, name, gain = 1.0):
        self.loss_trackers[name] = {
            'metric': tfk.metrics.Mean(name = name),
            'gain': gain
        }

    def call(self, inputs, *args, **kwargs):
        return self.network(inputs, *args, **kwargs)

    def _compute_loss(self, data):
        loss_total = 0
        # Model-specific losses
        result = self(data)
        if isinstance(result, dict):
            losses = result
        else:
            losses, outputs = result

        for k, v in losses.items():
            loss_total += self.loss_trackers[k]['gain'] * v

        # Layer losses (mostly regularization)
        reg_loss_total = 0
        for reg_loss in self.network.losses:
            reg_loss_total += reg_loss
        losses['reglr_loss'] = reg_loss_total
        loss_total += reg_loss_total
        # The total loss
        losses['total_loss'] = loss_total
        return losses, loss_total

    def set_training_plan(self, plan : nnu.TrainingPlan):
        self.training_plan = plan


    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            losses, loss_total = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss_total, self.network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        for k, v in self.loss_trackers.items():
            v['metric'].update_state(losses[k])
        return {k: v['metric'].result() for k, v in self.loss_trackers.items()}

    def test_step(self, data):
        losses, _ = self._compute_loss(data)
        # Let's update and return the loss metric.
        for k, v in self.loss_trackers.items():
            v['metric'].update_state(losses[k])
        return {k: v['metric'].result() for k, v in self.loss_trackers.items()}

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [v['metric'] for k, v in self.loss_trackers.items()]

class ModelManagerBase(abc.ABC):
    def __init__(self, seed : int = 1234, verbose : bool = False):
        self.seed = seed
        self.verbose = verbose
        self.model = None
        self.progress_tracker = None
        self.train_dataset = None
        self.val_dataset = None
        self.epochs_passed = 0
        self.history = None
        self.run_eagerly = False
        self.training_plan : nnu.TrainingPlan = None

    @abc.abstractmethod
    def build_model(self, **kwargs) -> tfk.Model:
        pass

    def do_train(self, dataset : tf.data.Dataset, n_samples, training_plan : nnu.TrainingPlan, **kwargs):
        tf.random.set_seed(self.seed)
        self.training_plan = training_plan
        progress_tracker = kwargs.pop('progress_tracker', True)
        if progress_tracker:
            self.progress_tracker = nnu.EpochProgressCallback(training_plan.epochs)
        else:
            self.progress_tracker = nnu.LoggingCallback()

        self.model = self.build_model(**kwargs)
        self.model.set_training_plan(training_plan)
        callbacks = [self.progress_tracker]
        training_log = training_plan.execute(self.model, dataset, n_samples, callbacks)
        self.epochs_passed = training_plan.epochs
        self.history = training_log.history

    def extend_train(self, epochs = 100, lr = 0.0001):
        self.progress_tracker.set_offset(self.epochs_passed)
        training_log = self.training_plan.execute_further(self.model, epochs = epochs, lr = lr)
        self.epochs_passed += epochs
        for k,v in training_log.history.items():
            self.history[k] += v

    def find_encoder(self, modality) -> tfk.Model:
        pass

    def find_decoder(self, modality) -> tfk.Model:
        pass

    def transform_to_common_space(self, adata_list : tp.List[sc.AnnData], unify = False, obs_fields = None):
        pass

    def predict_modality(self, adata : sc.AnnData, modality2 : str):
        pass

