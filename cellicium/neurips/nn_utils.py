from cellicium.utils import display
import tensorflow as tf
import tensorflow.keras as tfk
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

            if self.first_run_completed and (epoch % 20 == 0):
                self.plot_metrics()

        self.first_run_completed = True

    def on_train_end(self, logs=None):
        if self.interactive:
            self.pbar.close()
            self.plot_metrics()
            plt.close()