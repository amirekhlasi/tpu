import tensorflow as tf
import numpy as np
import estimator

tf.enable_eager_execution()

fake_train = np.random.uniform(-10, 10, size=[1024, 50]).astype(np.float32)
fake_dev = np.random.uniform(-10, 10, size=[128, 50]).astype(np.float32)


def _map(x):
    y = tf.reduce_sum(x, keep_dims=True) > 0
    y = tf.cast(y, tf.float32)
    return {"input": x, "label": y}


def data_fn():
    train_data = tf.data.Dataset.from_tensor_slices(fake_train)
    train_data = train_data.repeat()
    train_data = train_data.map(_map)
    train_data = train_data.batch(128, drop_remainder=True)
    dev_data = tf.data.Dataset.from_tensor_slices(fake_dev)
    dev_data = dev_data.repeat()
    dev_data = dev_data.map(_map)
    dev_data = dev_data.batch(128, drop_remainder=True)
    data_spec = estimator.DataSpec(train=train_data, dev=dev_data)
    return data_spec

def model_fn(data, training):
    layer1 = tf.keras.layers.Dense(50, "tanh")
    layer2 = tf.keras.layers.Dense(1, "sigmoid")
    dropout = 0.1 * tf.cast(training, tf.float32)
    x = data["input"]
    x = layer1(x)
    x = tf.nn.dropout(x, rate=dropout)
    x = layer2(x)
    y = data["label"]
    loss = tf.keras.losses.binary_crossentropy(y, x)
    loss = tf.reduce_mean(loss)
    accuracy = tf.keras.metrics.binary_accuracy(y, x)
    metric = {"accuracy": accuracy}
    optimizer = tf.train.GradientDescentOptimizer(1e-3)
    trainable_variables = layer1.weights + layer2.weights
    model_spec = estimator.ModelSpec(
        loss=loss,
        optimizer=optimizer,
        trainable_variables=trainable_variables,
        metric=metric
    )
    return model_spec

run_config = estimator.RunConfig(
    train_steps_per_round=1000,
    eval_steps_per_round=20,
    model_dir="model_dir",
    save_every_rounds=40,
)

estm = estimator.Estimator(model_fn, data_fn, run_config)

estm.run(200)




