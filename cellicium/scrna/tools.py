import numpy as np
import tensorflow as tf

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


def find_rotational_plane(X, V, n_comp = 10):
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
    output = tf.keras.layers.Dense(n_comp)(input)
    model = tf.keras.Model(input, output)
    model.compile(
        optimizer = 'adam',
        loss = tf.keras.losses.MeanSquaredError(),
        metrics = ['accuracy']
    )
    model.fit(X, V, epochs=150)
    return (model.layers[1].weights[0].numpy(), model.layers[1].weights[1].numpy())
