import tensorflow as tf

def generate_ann():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return model