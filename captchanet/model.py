import tensorflow as tf
from kerastuner import HyperParameters


def build_model(image_shape, image_type, vocabulary_size, max_len_word, params=None):

  def _builder(hp):

    kernel_size = (3, 3)
    pool_size = (2, 2)
    activation = 'relu'

    n_layers = hp.Range('n_layers', min_value=2, max_value=6, step=1, default=5)
    n_conv = hp.Range('n_conv', min_value=1, max_value=6, step=1, default=2)
    n_base_filters = hp.Choice('n_base_filters', values=[8, 16, 32], default=8)

    kernel_initializer = hp.Choice('kernel_initializer', values=['glorot_uniform', 'he_normal', 'he_uniform'], default='he_uniform')
    use_dense_layer = hp.Choice('use_dense_layer', values=['yes', 'no'], default='yes')
    dense_units = hp.Choice('dense_units', values=[512, 1024, 2048], default=2048)

    optimizer_name = hp.Choice('optimizer_name', values=['sgd', 'adam', 'rmsprop'], default='rmsprop')
    starting_lr = hp.Choice('starting_lr', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5], default=1e-3)
    momentum = hp.Choice('momentum', values=[0.9, 0.95, 0.99], default=0.9)

    tf.keras.backend.clear_session()

    inputs = tf.keras.layers.Input(name='image', shape=image_shape, dtype=image_type)
    x = inputs

    for i in range(n_layers):
      n_filters = n_base_filters * 2 ** min(i, 4)
      for _ in range(n_conv):
        x = tf.keras.layers.Conv2D(n_filters, kernel_size, padding='same', kernel_initializer=kernel_initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
      x = tf.keras.layers.MaxPooling2D(pool_size)(x)
      x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Flatten()(x)

    if use_dense_layer == 'yes':
      x = tf.keras.layers.Dense(dense_units)(x)
      x = tf.keras.layers.Activation(activation)(x)
      x = tf.keras.layers.Dropout(0.7)(x)

    outputs = []
    for i in range(max_len_word):
      out = tf.keras.layers.Dense(vocabulary_size, activation='softmax', name=f'character_{i}')(x)
      outputs.append(out)
    outputs = tf.keras.layers.Concatenate()(outputs)
    outputs = tf.keras.layers.Reshape((max_len_word, vocabulary_size))(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Optimizer Parameters
    sgd_params = {}
    sgd_params['learning_rate'] = starting_lr
    sgd_params['momentum'] = momentum
    sgd_params['nesterov'] = True

    adam_params = {}
    adam_params['learning_rate'] = starting_lr
    adam_params['amsgrad'] = True

    rmsprop_params = {}
    rmsprop_params['learning_rate'] = starting_lr

    # Build optimizer.
    if optimizer_name == 'sgd':
      optimizer = tf.keras.optimizers.SGD(**sgd_params)
    elif optimizer_name == 'adam':
      optimizer = tf.keras.optimizers.Adam(**adam_params)
    elif optimizer_name == 'rmsprop':
      optimizer = tf.keras.optimizers.RMSprop(**rmsprop_params)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

  hp = HyperParameters()

  if params:
    for key, value in params.items():
      hp.Fixed(key, value)

  return _builder(hp)
