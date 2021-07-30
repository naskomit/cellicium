import numpy as np
import tensorflow as tf
import scipy.optimize as spopt

def add_intron_data(adata, intron_data):
  assert(adata.n_obs == intron_data.n_obs, "Number of observations must be the same for both exonic and intronic reads")
  #exon_genes = exon
  intron_layer = np.zeros(adata.X.shape)
  for i, v in enumerate(adata.var.index):
      #print(i, v)
      if v in intron_data.var.index:
          intron_layer[:, i] = intron_data[:, v].X.flatten()
  adata.layers['spliced'] = adata.X
  adata.layers['unspliced'] = intron_layer
  return adata


def fit_matrix(X, V, n_comp = 10, use_bias = True):
    print("X: ", X.shape, "V: ", V.shape)
    X = X[:, :n_comp]
    V = V[:, :n_comp]
    n_points = X.shape[0]
    # x_in = tf.placeholder("float", [None, n_comp])
    # v_out = tf.placeholder("float", [None, n_comp])
    # W = tf.Variable(tf.zeros(n_comp, n_comp))
    # b = tf.Variable(tf.zeros([n_comp]))
    # print(X.shape)
    # print(V.shape)
    input = tf.keras.Input(shape = (n_comp,))
    output = tf.keras.layers.Dense(n_comp, use_bias = use_bias)(input)
    model = tf.keras.Model(input, output)
    model.compile(
        optimizer = 'adam',
        loss = tf.keras.losses.MeanSquaredError(),
        metrics = ['accuracy']
    )
    training = model.fit(X, V, epochs=300, verbose = False)
    final_loss = training.history['loss'][-1]
    final_accuracy = training.history['accuracy'][-1]
    print(f'Final loss: {final_loss}')
    print(f'Final accuracy: {final_accuracy}')
    V_pred = model.predict(X)
    result = {
        'W': model.layers[1].weights[0].numpy(),
        'V_pred': V_pred,
        'training': training,
        'model': model
    }
    if use_bias:
        result['b'] = model.layers[1].weights[1].numpy()
    return result

def fit_multilayer(X, V, n_layers = 2):
    n_points = X.shape[0]
    n_comp = X.shape[1]
    use_bias = True
    activation = tf.keras.activations.tanh
    input = tf.keras.Input(shape = (n_comp,))
    hidden = input
    for i in range(n_layers):
        hidden = tf.keras.layers.Dense(n_comp, use_bias = use_bias, activation = activation)(hidden)
    output = hidden
    model = tf.keras.Model(input, output)
    model.compile(
        optimizer = 'adam',
        loss = tf.keras.losses.MeanSquaredError(),
        metrics = ['cosine_similarity']
    )
    training = model.fit(X, V, epochs=300, verbose = False)
    print_metrics = ['loss', 'cosine_similarity']
    for item in print_metrics:
        initial_value = training.history[item][0]
        print(f'Initial {item}: {initial_value}')
    for item in print_metrics:
        final_value = training.history[item][-1]
        print(f'Final {item}: {final_value}')
    V_pred = model.predict(X)
    result = {
        'V_pred': V_pred,
        'training': training,
        'model': model
    }
    # if use_bias:
    #     result['b'] = model.layers[1].weights[1].numpy()
    return result


def bin_smooth(t, X, n_bins = 50):
    dt = (t[-1] - t[0])  / n_bins
    X_smooth = np.zeros((n_bins, X.shape[1]))
    for i in range(n_bins):
        t_start = i * dt
        t_end = (i + 1) * dt
        bin_filter = (t >= t_start) & (t < t_end)
        X_smooth[i, :] = np.mean(X[bin_filter, :], axis = 0)
    return (np.linspace(0, t[-1], n_bins), X_smooth)


def fit_peak(data):
    n_points = data.shape[0]
    def f1(x):
        mu = x[0]
        sigma = x[1]
        A = x[2]
        B = x[3]
        z = np.linspace(0, 1, n_points)
        y = A * np.exp(-((z - mu) / sigma)**2) + B
        return y

    def f1_loss(data_in):
        return lambda x: np.linalg.norm(f1(x) - data_in)

    res = spopt.minimize(
        f1_loss(data), x0 = [0.5, 0.5, 2, 0],
        bounds = [[0, 1], [0.1, 1], [0, 10], [0, 10]])
    return (f1(res.x), res)
