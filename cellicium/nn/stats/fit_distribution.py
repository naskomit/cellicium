import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp

def fit_distribution(dist : tfp.distributions.Distribution, data : np.ndarray):
    num_params = 2
