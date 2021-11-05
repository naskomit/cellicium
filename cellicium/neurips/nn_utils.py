from cellicium.utils import display
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def ensure_dense(X):
    return X.todense() if sparse.issparse(X) else X


class LinLogLayer(tfk.layers.Layer):
    def __init__(self, shape, log_offset, name = None):
        super().__init__(name = name)
        self.shape = shape
        self.log_offset = log_offset

    def build(self, input_shape):
        self.lin_dense = tfk.layers.Dense(units = self.shape)
        self.log_dense = tfk.layers.Dense(units = self.shape)

    def call(self, input, **kwargs):
        # if shape is None:
        #     shape = input.shape[-1]
        #, kernel_regularizer = tfk.regularizers.l1_l2(l1_reg, l2_reg)
        x_lin = self.lin_dense(input)
        x_log = tf.math.log(input + self.log_offset)
        x_log = self.log_dense(x_log)
        x_log = tf.math.exp(x_log)
        x = x_lin + x_log
        return x

class EpochProgressCallback(tfk.callbacks.Callback):
    def __init__(self, total_num_epochs : int):
        from tqdm.notebook import tqdm
        super().__init__()
        self.total_num_epochs = total_num_epochs
        self.interactive = False
        self.epoch_offset = 0
        try:
            import ipywidgets as widgets
            self.interactive = True
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

            training_output = widgets.HBox([widgets.VBox([progress_bar_out, self.metrics_out]), self.fig_out],
                                           layout={'border': '1px solid black', 'height': '300px'})
            display(training_output)

            with progress_bar_out:
                self.pbar = tqdm(total = total_num_epochs)
        except ModuleNotFoundError:
            pass
        # self.fig = plt.figure()
        self.metrics = {}
        self.first_run_completed = False

    def set_offset(self, value):
        self.epoch_offset = value

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
        if self.interactive:
            self.pbar.update(1)

            if not self.first_run_completed:
                for k, v, in logs.items():
                    self.metrics[k] = []

            for k, v, in logs.items():
                self.metrics[k].append(v)
                self.metrics_out.clear_output()
                with self.metrics_out:
                    print(f'Epoch {epoch + 1 + self.epoch_offset}')
                    for k, v, in logs.items():
                        print(f'{k}: {v:.3e}')

            if self.first_run_completed and (epoch % 10 == 0):
                self.plot_metrics()

        self.first_run_completed = True

    def on_train_end(self, logs=None):
        if self.interactive:
            self.pbar.close()
            self.plot_metrics()
            plt.close()


class TrainingPlan(tfk.callbacks.Callback):
    def __init__(self, epochs : int, lr : float, minibatch_size :int = 256,
                 kl_warmup : float = None, kl_weight : float = 1e-3, weight_decay = 1e-6,
                 validation_split : float = 0.0, run_eagerly = False):
        self.epochs = epochs
        self.lr = lr
        self.minibatch_size = minibatch_size
        self.kl_warmup = kl_warmup
        self.kl_weight = kl_weight
        self.weight_decay = weight_decay
        self.validation_split = validation_split
        self.run_eagerly = run_eagerly

        self.train_dataset : tf.data.Dataset = None
        self.val_dataset : tf.data.Dataset = None
        self.extra_callbacks = []

        self._current_epoch = tf.Variable(0, dtype = tf.int32, trainable = False)

    @property
    def current_epoch(self):
        return self._current_epoch

    def on_epoch_begin(self, epoch, logs = None):
        self._current_epoch.assign_add(1)

    def current_kl_weight(self):
        if self.kl_warmup is not None:
            k = tf.math.minimum(1.0, tf.cast(self.current_epoch / self.kl_warmup, tf.float32))
        else:
            k = 1.0
        return k * self.kl_weight

    def execute(self, model, dataset, n_samples, callbacks, **kwargs):
        n_train = round(n_samples * (1 - self.validation_split))
        print(f'Number of train/validation samples: {n_train} / {n_samples - n_train}')
        dataset = dataset.shuffle(buffer_size = 8192)
        train_dataset = dataset.take(n_train)
        val_dataset = dataset.skip(n_train)

        if self.minibatch_size is not None:
            train_dataset = train_dataset.batch(self.minibatch_size, drop_remainder=False)
            val_dataset = val_dataset.batch(self.minibatch_size, drop_remainder=False)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # weight_decay = self.weight_decay,
        model.compile(optimizer = tf.optimizers.Adam(
            learning_rate = self.lr),
            run_eagerly = self.run_eagerly)

        self.extra_callbacks = callbacks
        callbacks = [self, *self.extra_callbacks]

        training_log = model.fit(
            train_dataset, epochs = self.epochs, validation_data = self.val_dataset,
            verbose = False, callbacks = callbacks
        )

        return training_log

    def execute_further(self, model, epochs : int, lr : float = None):
        if (self.train_dataset is None) or (self.val_dataset is None):
            raise RuntimeError('Cannot extend training if no initial training is performed')
        if lr is not None:
            self.lr = lr
        callbacks = [self, *self.extra_callbacks]
        model.compile(optimizer = tfa.optimizers.AdamW(
            weight_decay = self.weight_decay, learning_rate = self.lr),
            run_eagerly = self.run_eagerly)
        training_log = model.fit(
            self.train_dataset, epochs = epochs, validation_data = self.val_dataset,
            verbose = False, callbacks = callbacks
        )

        return training_log


