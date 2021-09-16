import numpy as np
import tensorflow as tf
import scipy.optimize as spopt

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
        metrics = ['cosine_similarity']
    )
    training = model.fit(X, V, epochs = 300, verbose = False)
    loss = training.history['loss']
    cosine_similarity = training.history['cosine_similarity']
    print(f'Initial loss: {loss[0]}')
    print(f'Initial accuracy: {cosine_similarity[0]}')
    print(f'Final loss: {loss[-1]}')
    print(f'Final accuracy: {cosine_similarity[-1]}')
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
