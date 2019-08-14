import tensorflow as tf
import estimator
from gpt2 import GPT2
import os
import json

tf.enable_eager_execution()

# Model in eager mode
model_path = "model"
with open(os.path.join(model_path, "hparams.json")) as f:
    config = json.load(f)
model = GPT2(config, name="gpt2")
x = tf.zeros([0, 0], dtype=tf.int32)
_ = model(x) # build model

model.load_weights(os.path.join(model_path, "weights.h5"))

def _data_builder(file_path, batch_size, pad_size):
    data = tf.data.TextLineDataset(file_path)
    data = data.repeat()

    def _map(x):
        x = tf.expand_dims(x, 0)
        tokens = tf.strings.split(x, " ").values
        tokens = tf.strings.to_number(tokens, tf.int32)
        length = tf.shape(tokens)[0]
        return {"tokens": tokens, "length": length}

    data = data.map(_map)
    output_shape = {"tokens": tf.TensorShape([pad_size]), "length": tf.TensorShape([])}
    data = data.padded_batch(batch_size, output_shape)
    return data

def data_fn():
    data_path = "data"
    train = _data_builder(os.path.join(data_path, "train.txt"), 8, 1025)
    dev = _data_builder(os.path.join(data_path, "test.txt"), 8, 1025)
    data_spec = estimator.DataSpec(train=train, dev=dev)
    return data_spec

def model_fn(data, training):
    model = GPT2(config, name="gpt2")
    inputs = data["tokens"][:, :-1]
    labels = data["tokens"][:, 1:]
    dropout = tf.cast(training, tf.float32) * 0.05
    logits = model(inputs, use_2d=True, attention_dropout=dropout, dropout=dropout)
    labels = tf.reshape(labels, [-1])
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    mask = tf.sequence_mask(data["length"] - 1, maxlen=labels.shape[1])
    mask = tf.reshape(mask, [-1])
    mask = tf.cast(mask, loss.dtype)
    loss = tf.reduce_sum(mask * loss) / tf.reduce_sum(mask)
    lr = tf.Variable(1e-4, name="lr")
    model_spec = estimator.ModelSpec(
        loss=loss,
        optimizer=tf.train.GradientDescentOptimizer(lr),
        trainable_variables=model.weights,
        import_variables=model.weights
    )
    return model_spec

run_config = estimator.RunConfig(
    train_steps_per_round=200,
    eval_steps_per_round=10,
    model_dir="model",
)


estm = estimator.Estimator(model_fn, data_fn, run_config)


values = [v.numpy() for v in model.weights]
estm.import_variables(values)

estm.run(200)

values = estm.export_model()
for u, v in zip(values, model.weights):
    v.assign(u)
model.save_weights(os.path.join(model_path, "new_weigths.h5"))


