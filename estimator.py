import tensorflow as tf
import numpy as np
import os
from google.colab import drive


class _utils(object):
    @staticmethod
    def flatten(structure):
    
        def _flatten(structure, index):
            if isinstance(structure, list):
                keys = []
                values = []
                for _str in structure:
                    _key, _value, index = _flatten(_str, index)
                    keys.append(_key)
                    values = values + _value
                return keys, values, index
            elif isinstance(structure, tuple):
                keys = []
                values = []
                for _str in structure:
                    _key, _value, index = _flatten(_str, index)
                    keys.append(_key)
                    values = values + _value
                keys = tuple(keys)
                return keys, values, index
            elif isinstance(structure, dict):
                keys = []
                values = []
                for key, _str in structure.items():
                    _key, _value, index = _flatten(_str, index)
                    keys.append((key, _key))
                    values = values + _value
                keys = dict(keys)
                return keys, values, index
            else:
                return index, [structure], index + 1
    
        keys, values, _ = _flatten(structure, 0)
        return keys, values
    
    @staticmethod
    def reconstruct(keys, values):
        if isinstance(keys, list):
            structure = []
            for _key in keys:
                _str = _utils.reconstruct(_key, values)
                structure.append(_str)
            return structure
        elif isinstance(keys, tuple):
            structure = []
            for _key in keys:
                _str = _utils.reconstruct(_key, values)
                structure.append(_str)
            structure = tuple(structure)
            return structure
        elif isinstance(keys, dict):
            structure = []
            for key, _key in keys.items():
                _str = _utils.reconstruct(_key, values)
                structure.append((key, _str))
            structure = dict(structure)
            return structure
        else:
            return values[keys]
    
    @staticmethod
    def get_tensor_shape(tensor):
        tensor = tf.convert_to_tensor(tensor)
        static_shape = tensor.shape.as_list()
        if tf.executing_eagerly():
            return static_shape
        dynamic_shape = tf.shape(tensor)
        if static_shape is None:
            return dynamic_shape
        dynamic_shape = tf.unstack(dynamic_shape)
        shape = []
        for st, dyn in zip(static_shape, dynamic_shape):
            if st is None:
                shape.append(dyn)
            else:
                shape.append(st)
        return shape



class ModelSpec(object):
    """
    Build the Model Specifications retunred by model_fn including:
        loss: a scalar tensor of type tf.float32.
        optimizer: an object of class tf.train.Optimizer, used for updating trainable_variables.
        trainable_variables: a list of variables to be trained by optimizer.
        metric: (Optional) a dictionary of metrics to be evaluated (like accuracy, ...). the values must be scalar tensors of type tf.float32
        global_step: (Optional) a Variable to be updated after each step of training. it is usually used for learning rate schedule.
        import_variables: (Optional) a list of variables which their values will be imported in the beginning of training.
    """
    def __init__(self, loss, optimizer, trainable_variables, metric=None,
                 global_step=None, import_variables=None):
        if loss.dtype != tf.float32:
            raise ValueError("loss must have dtype tf.float32")
        if loss.shape.as_list() != []:
            raise ValueError("loss should be a scalar")
        self.loss = loss
        if not isinstance(optimizer, tf.train.Optimizer):
            raise ValueError("optimizer must an instance of tf.train.Optimizer")
        self.optimizer = optimizer
        if not isinstance(trainable_variables, list):
            raise ValueError("trainable_variables must be a list of variables")
        self.trainable_variables = trainable_variables
        if metric is not None:
            if not isinstance(metric, dict):
                raise ValueError("metric must be a dictionary with scalar tensors values")
            for tensor in metric.values():
                if tensor.dtype != tf.float32:
                    raise ValueError("metric tensors must have dtype tf.float32")
                if tensor.shape.as_list() != []:
                    raise ValueError("metric tensors must be scalar")
        else:
            metric = dict()
        self.metric = metric
        if global_step is not None:
            if not isinstance(global_step, tf.Variable):
                raise ValueError("global step must be a variable")
            if global_step.dtype != tf.int64:
                raise ValueError("global step must have type tf.int64")
            if global_step.shape.as_list() != []:
                raise ValueError("global step must be scalar")
        self.global_step = global_step
        if import_variables is not None:
            if not isinstance(import_variables, list):
                raise ValueError("import_variables must be a list of variables")
        self.import_variables = import_variables

    @staticmethod
    def from_dict(dictionary):
        if not isinstance(dictionary, dict):
            raise ValueError("dictionary must be a dict")
        inputs = ["loss", "optimizer", "trainable_variables", "metric", "import_variables", "global_step"]
        for x in dictionary.keys():
            if x not in inputs:
                raise ValueError("ModelSpec does not accept {} as input".format(x))
        model_spec = ModelSpec(**dictionary)
        return model_spec


class DataSpec(object):
    """
    Build the Data specification for estimator returned by data_fn including:
        train: dataset for train.
        dev: dataset for test.
    Both of them should be object of class tf.data.Dataset.
    The output of datasets should be a nested structure(tuples, list, dictionary) of tensors. static shape of
    all tensors should be fully definitive and same for train and dev.
    The first dimension of dataset outputs would be considered as batch size and should be divisible by tpu cores numbers.
    Note: the exterior structure of dataset output should not be tuple.
    """
    def __init__(self, train, dev):
        if not isinstance(train, tf.data.Dataset):
            raise ValueError("train must be instance of tf.data.Data")
        if not isinstance(dev, tf.data.Dataset):
            raise ValueError("dev must be instance of tf.data.Data")
        key_maps, values = _utils.flatten(train.output_shapes)
        for v in values:
            if not v.is_fully_defined():
                raise ValueError("all outputs shapes must be fully defined")
        if train.output_types != dev.output_types:
            raise ValueError("train and dev outputs must have same nested structure and dtype")
        if train.output_shapes != dev.output_shapes:
            raise ValueError("train and dev outputs must have same shapes")
        if isinstance(key_maps, tuple):
            raise ValueError("the exterior structure of data output should not be tuple")
        self.key_maps = key_maps
        self.output_shapes = train.output_shapes
        self.train = train
        self.dev = dev

    @staticmethod
    def from_dict(dictionary):
        if not isinstance(dictionary, dict):
            raise ValueError("dictionary must be instance of dict")
        inputs = ["train", "dev"]
        for key in dictionary.keys():
            if key not in inputs:
                raise ValueError("DataSpec does not accept {}".format(key))
        data_spec = DataSpec(** dictionary)
        return data_spec

class RunConfig(object):
    """
    Run Configuration for Estimator.
    The training loop consist of rounds. In each round you have some steps of training and some steps of evaluation.
    the configuration includes:
        train_steps_per_round: training steps for each round.
        eval_steps_per_round: evaluation steps for each round.
        model_dir: A directory for saving checkpoints. If the directory is not empty,
                   the model will start training from the latest checkpoint.
        save_every_rounds: the model will save checkpoint for every "save_every_rounds" round.
        restore_data_state: if True, the model will restore the latest data state
        checkpoint_max_keep: maximum number of checkpoints too keep.
        num_cores: the numbers of tpu cores.
        drive_path: if it is specified, then the system will flush and remount drive connection after saving each checkpoint
                    in order to avoid problems with drive syncnorization.
        WARNING: if you are using drive_path option, do not read the data from drive directly, instead copy data to local memory of colab.
                 also your current working path should not be a child of drive path.
    """
    def __init__(self, train_steps_per_round, eval_steps_per_round,
                 model_dir, save_every_rounds=20, restore_data_state=True,
                 checkpoint_max_keep=5, num_cores=8, drive_path=None):
        self.train_steps_per_round = train_steps_per_round
        self.eval_steps_per_round = eval_steps_per_round
        self.model_dir = model_dir
        self.save_every_rounds = save_every_rounds
        self.restore_data_state = restore_data_state
        self.checkpoint_max_keep = checkpoint_max_keep
        self.num_cores = num_cores
        self.drive_path = drive_path

    @staticmethod
    def from_dict(dictionary):
        if not isinstance(dictionary, dict):
            raise ValueError("dictionary must be instance of dict")
        inputs = [
            "train_steps_per_round",
            "eval_steps_per_round",
            "model_dir",
            "save_every_rounds",
            "restore_data_state",
            "checkpoint_max_keep",
            "num_cores",
            "drive_path"
        ]
        for key in dictionary.keys():
            if key not in inputs:
                raise ValueError("DataSpec does not accept {}".format(key))
        run_config = RunConfig(**dictionary)
        return run_config

class Estimator(object):
    """
    Estimator is a high level API for running on Colab TPU. The main logic of this class is inspired by
    the main Tensorflow Estimators, but it is different in some aspects and more simple. The main difference is that this Estimator
    will create the whole computation graph in initialization and it has different training loop (see RunConfig).

    The Estimator can only be used when the training loop is an ordinary supervised or unsupervised training loop:
    getting  inputs, computing the loss, computing gradients and updating trainable weights.

    Estimator needs three arguments to be built:
        model_fn: a callable object which returns a ModelSpec object. The model_fn will build the main part of computational graph for
                  training step. All Tensorflow operations should be defined in model_fn and consider that this object
                  will be called twice (one for tpu cloud VM and one for local Colab VM). For more information about model_fn
                  see ModelSpec.
        data_fn: a callable object which returns a DataSpec object. It will build the data pipeline and like model_fn it
                 all Tensorflow operations for data pipeline should be defined in data_fn.
                 For more information about data_fn see DataSpec
        run_config: a RunConfig object which specify the run time configurations. See RunConfig


    Notes:
          * The model_fn and data_fn will be called twice: one for building operational graph for local Colab VM and one for
            building operational graph for Cloud TPU VM.
          * The environment should be eager.

    """
    def __init__(self, model_fn, data_fn, run_config):
        if not callable(model_fn):
            raise ValueError("model_fn should be callable")
        if not callable(data_fn):
            raise ValueError("data_fn should be callable")
        if not isinstance(run_config, RunConfig):
            raise ValueError("run_config should be instance of RunConfig")
        if 'COLAB_TPU_ADDR' not in os.environ:
            raise OSError("Colab TPU not found. check the notebook options")
        self._run_config = run_config
        tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
        self._cpu_graph = tf.Graph()
        self._tpu_graph = tf.Graph()
        self._build_dataset(data_fn)
        self._build_cpu_model(model_fn)
        self._build_tpu_model(model_fn)
        self._cpu_session = tf.Session(graph=self._cpu_graph)
        self._tpu_session = tf.Session(target=tpu_address, graph=self._tpu_graph)
        self._round = 0
        self._build_graph()
        self._build_restore_data_graph()
        with self._cpu_graph.as_default():
            self._cpu_session.run(tf.global_variables_initializer())
        with self._tpu_graph.as_default():
            self._tpu_session.run(tf.tpu.initialize_system())
            self._tpu_session.run(tf.global_variables_initializer())
        self._create_checkpoint()
        self._sync_up()

    def _build_dataset(self, data_fn):
        with self._cpu_graph.as_default():
            _data_spec = data_fn()
            if not isinstance(_data_spec, DataSpec):
                raise ValueError("data_fn must return a DataSpec")

            def _map(inputs):
                keys, values = _utils.flatten(inputs)
                new_values = []
                for v in values:
                    shape = v.shape.as_list()
                    n = len(shape)
                    transpose = [1, 0] + list(range(2, n + 1))
                    w = tf.reshape(v, [shape[0], self._run_config.num_cores,
                                       shape[1] // self._run_config.num_cores] + shape[2:])
                    w = tf.transpose(w, transpose)
                    new_values.append(w)
                return new_values

            self._data_spec = _data_spec
            self._train_data = self._data_spec.train.batch(self._run_config.train_steps_per_round, True)
            self._dev_data = self._data_spec.dev.batch(self._run_config.eval_steps_per_round, True)
            self._train_data = self._train_data.map(_map)
            self._dev_data = self._dev_data.map(_map)

    def _build_cpu_model(self, model_fn):
        with self._cpu_graph.as_default():
            dtypes = self._data_spec.train.output_types
            shapes = self._data_spec.train.output_shapes
            _, dtypes = _utils.flatten(dtypes)
            _, shapes = _utils.flatten(shapes)
            placeholders = [tf.placeholder(d, s) for d, s in zip(dtypes, shapes)]
            placeholders = _utils.reconstruct(self._data_spec.key_maps, placeholders)
            _training = tf.placeholder(tf.bool, [])
            self._cpu_model_spec = model_fn(placeholders, _training)
            if not isinstance(self._cpu_model_spec, ModelSpec):
                raise ValueError("model_fn must return a ModelSpec")
            grads = tf.gradients(self._cpu_model_spec.loss, self._cpu_model_spec.trainable_variables)
            grads_and_vars = zip(grads, self._cpu_model_spec.trainable_variables)
            opt = self._cpu_model_spec.optimizer.apply_gradients(grads_and_vars, self._cpu_model_spec.global_step)
            self._cpu_variables = {v.name: v for v in tf.global_variables()}
            self._cpu_variables_plc = {v.name: tf.placeholder(v.dtype, v.shape) for v in tf.global_variables()}
            self._cpu_variables_assign = {v.name: v.assign(plc) for v, plc in
                                          zip(tf.global_variables(), self._cpu_variables_plc.values())}
            self._variables_name = list(self._cpu_variables.keys())
            self._metric_keys, _ = _utils.flatten(self._cpu_model_spec.metric)
            self._train_step = tf.Variable(0, dtype=tf.int64, name="train_step")
            self._dev_step = tf.Variable(0, dtype=tf.int64, name="eval_step")

    def _build_restore_data_graph(self):
        def _restore(steps, iterator):
            def body(step):
                with tf.control_dependencies(iterator.get_next()):
                    step = step + 1
                return step

            def cond(step):
                return step < steps
            op = tf.while_loop(cond, body, [tf.zeros([], tf.int64)], parallel_iterations=1)
            return op
        with self._cpu_graph.as_default():
            steps = self._train_step // self._run_config.train_steps_per_round
            op1 = _restore(steps, self._train_iterator)
            steps = self._dev_step // self._run_config.eval_steps_per_round
            op2 = _restore(steps, self._dev_iterator)
            self._restore_data_op = tf.group(op1, op2)


    def _build_tpu_model(self, model_fn):

        def train_step(training, inputs):
            inputs = _utils.reconstruct(self._data_spec.key_maps, inputs)
            self._tpu_model_spec = model_fn(inputs, training)
            if not isinstance(self._tpu_model_spec, ModelSpec):
                raise ValueError("model_fn must return a ModelSpec")
            loss = self._tpu_model_spec.loss
            optimizer = tf.tpu.CrossShardOptimizer(self._tpu_model_spec.optimizer)
            def true_fn():
                weights = self._tpu_model_spec.trainable_variables
                grads = tf.gradients(loss, weights)
                opt = optimizer.apply_gradients(zip(grads, weights), self._tpu_model_spec.global_step)
                with tf.control_dependencies([opt]):
                    opt = tf.zeros([], tf.bool)
                return opt

            def false_fn():
                opt = tf.zeros([], tf.bool)
                return opt

            opt = tf.cond(training, true_fn, false_fn)
            returns = [loss, opt]
            metric_keys, metric_values = _utils.flatten(self._tpu_model_spec.metric)
            returns = returns + metric_values
            return returns

        def train(training, *data):
            training = tf.squeeze(training)
            data = [tf.squeeze(d, 0) for d in data]
            total_steps = tf.shape(data[0])[0]

            def body(step, loss, *metrics):
                datapoint = [tf.gather(d, step) for d in data]
                result = train_step(training, datapoint)
                _loss, opt, _metrics = result[0], result[1], result[2:]
                loss = _loss + loss
                metrics = [m + _m for m, _m in zip(metrics, _metrics)]
                with tf.control_dependencies([opt]):
                    step = step + 1
                result = [step, loss] + metrics
                return tuple(result)

            def cond(step, loss, *metrics):
                return step < total_steps

            step = tf.zeros([], tf.int32)
            loss = tf.zeros([])
            metrics = [tf.zeros([]) for _ in self._cpu_model_spec.metric]
            variables = [step, loss] + metrics
            result = tf.contrib.tpu.while_loop(cond, body, variables)
            loss, metrics = result[1], result[2:]
            loss = loss / tf.cast(total_steps, tf.float32)
            metrics = [m / tf.cast(total_steps, tf.float32) for m in metrics]
            returns = [loss] + metrics
            return tuple(returns)

        with self._tpu_graph.as_default():
            self._tpu_training = tf.placeholder(tf.bool, [])
            types = self._train_data.output_types
            _shapes = self._train_data.output_shapes
            shapes = []
            for shape in _shapes:
                _shape = shape.as_list()
                _shape = [_shape[0], None] + _shape[2:]
                shapes.append(_shape)
            self._tpu_inputs_plc = [tf.placeholder(d, s) for d, s in zip(types, shapes)]
            training = tf.fill([self._run_config.num_cores], self._tpu_training)
            inputs = [training] + self._tpu_inputs_plc
            results = tf.tpu.shard(train, inputs, self._run_config.num_cores)
            results = [tf.reduce_mean(res) for res in results]
            self._tpu_loss = results[0]
            metrics = results[1:]
            self._tpu_metrics = dict(zip(self._cpu_model_spec.metric.keys(), metrics))
            self._tpu_variables = {name: v for name, v in zip(self._variables_name, tf.global_variables())}
            self._tpu_variables_plc = {name: tf.placeholder(v.dtype, v.shape) for name, v in self._cpu_variables.items()}
            self._tpu_variables_assign = {name: v.assign(u)
                                          for name, u, v in zip(self._variables_name, self._tpu_variables_plc.values(),
                                                                self._tpu_variables.values())}

    def _sync_up(self, var_names=None):
        if var_names is None:
            var_names = self._variables_name
        if not isinstance(var_names, list):
            raise ValueError("var_names must be a list")
        cpu_vars = [self._cpu_variables[name] for name in var_names]
        tpu_vars_assign = [self._tpu_variables_assign[name] for name in var_names]
        tpu_vars_plc = [self._tpu_variables_plc[name] for name in var_names]
        vars_value = self._cpu_session.run(cpu_vars)
        feed_dict = {plc: value for plc, value in zip(tpu_vars_plc, vars_value)}
        self._tpu_session.run(tpu_vars_assign, feed_dict=feed_dict)

    def _sync_down(self, var_names=None):
        if var_names is None:
            var_names = self._variables_name
        if not isinstance(var_names, list):
            raise ValueError("var_names must be a dictionary")
        tpu_vars = [self._tpu_variables[name] for name in var_names]
        cpu_vars_assign = [self._cpu_variables_assign[name] for name in var_names]
        cpu_vars_plc = [self._cpu_variables_plc[name] for name in var_names]
        vars_value = self._tpu_session.run(tpu_vars)
        feed_dict = {plc: value for plc, value in zip(cpu_vars_plc, vars_value)}
        self._cpu_session.run(cpu_vars_assign, feed_dict=feed_dict)

    def _run_round(self, *inputs):
        ln = len(inputs) // 2
        train_inputs = inputs[:ln]
        dev_inputs = inputs[ln:]
        train_inputs = [_input.numpy() for _input in train_inputs]
        dev_inputs = [_input.numpy() for _input in dev_inputs]
        operations = [self._tpu_loss, self._tpu_metrics]
        feed_dict = dict(zip(self._tpu_inputs_plc, train_inputs))
        feed_dict[self._tpu_training] = True
        train_result = self._tpu_session.run(operations, feed_dict)
        feed_dict = dict(zip(self._tpu_inputs_plc, dev_inputs))
        feed_dict[self._tpu_training] = False
        dev_result = self._tpu_session.run(operations, feed_dict)
        self._round += 1
        print("round: ", self._round)
        print("train: ")
        print("loss: ", train_result[0])
        for key, value in train_result[1].items():
            print(key + ": ", value)
        print("eval: ")
        print("loss: ", dev_result[0])
        for key, value in dev_result[1].items():
            print(key + ": ", value)
        print("\n\n")
        return tf.zeros([], tf.bool)

    def _build_graph(self):
        with self._cpu_graph.as_default():
            self._num_round = tf.placeholder(tf.int32, [])
            self._train_iterator = self._train_data.make_one_shot_iterator()
            self._dev_iterator = self._dev_data.make_one_shot_iterator()

            def _get_inputs(fake=False):
                def true_fn():
                    train = [tf.zeros(shape, type) for shape, type in
                              zip(self._train_data.output_shapes, self._train_data.output_types)]
                    dev = [tf.zeros(shape, type) for shape, type in
                              zip(self._dev_data.output_shapes, self._dev_data.output_types)]
                    return tuple(train + dev)
                def false_fn():
                    output = self._train_iterator.get_next() + self._dev_iterator.get_next()
                    return output
                fake = tf.convert_to_tensor(fake)
                inputs = tf.cond(fake, true_fn, false_fn)
                return inputs

            def body(round, inputs):
                flag = tf.py_function(self._run_round, inputs, tf.bool)
                fake = tf.equal(round, self._num_round - 1)
                inputs = _get_inputs(fake)
                with tf.control_dependencies([flag]):
                    round = round + 1
                return round, inputs

            def cond(round, inputs):
                return round < self._num_round

            round = tf.zeros([], tf.int32)
            fake = tf.equal(round, self._num_round)
            inputs = _get_inputs(fake)
            self._run_opt, _ = tf.while_loop(cond, body, [round, inputs], parallel_iterations=1)

    def _create_checkpoint(self):
        with self._cpu_graph.as_default():
            with self._cpu_session.as_default():
                self._cpu_var_ckpt = self._cpu_model_spec.trainable_variables + self._cpu_model_spec.optimizer.variables()
                self._tpu_var_ckpt = self._tpu_model_spec.trainable_variables + self._tpu_model_spec.optimizer.variables()
                if self._cpu_model_spec.global_step is not None:
                    self._cpu_var_ckpt.append(self._cpu_model_spec.global_step)
                    self._tpu_var_ckpt.append(self._tpu_model_spec.global_step)
                self._cpu_var_ckpt_names = [v.name for v in self._cpu_var_ckpt]
                ckpt = dict(zip(self._cpu_var_ckpt_names, self._cpu_var_ckpt))
                ckpt[self._train_step.name] = self._train_step
                ckpt[self._dev_step.name] = self._dev_step
                self._checkpoint = tf.train.Checkpoint(**ckpt)
                path = os.path.join(self._run_config.model_dir, "checkpoints")
                self._checkpoint_manager = tf.train.CheckpointManager(self._checkpoint, path,
                                                                      max_to_keep=self._run_config.checkpoint_max_keep)
        if self._checkpoint_manager.latest_checkpoint is not None:
            self.restore(self._checkpoint_manager.latest_checkpoint, self._run_config.restore_data_state)

    def _save_checkpoint(self):
        self._sync_down(self._cpu_var_ckpt_names)
        with self._cpu_graph.as_default():
            with self._cpu_session.as_default():
                self._checkpoint_manager.save()
        if self._run_config.drive_path is not None:
            drive.flush_and_unmount()
            drive.mount(self._run_config.drive_path)

    def get_variable_names(self):
        """
        Returns a list of all variable names used in computational graph
        """
        return self._variables_name.copy()

    def set_variables_value(self, name_value_dict):
        """
        the value of variables will be set as name_value_dict
        Args:
            name_value_dict: a dictionary of {variable name: variable value}
        """
        feed_dict = {self._cpu_variables_plc[key]: value for key, value in name_value_dict.items()}
        for key, value in feed_dict.items():
            if key.shape.as_list() != list(np.array(value).shape):
                raise ValueError("values shapes and variables shapes do not match")
        operations = [self._cpu_variables_assign[key] for key in name_value_dict.keys()]
        self._cpu_session.run(operations, feed_dict)
        feed_dict = {self._tpu_variables_plc[key]: value for key, value in name_value_dict.items()}
        operations = [self._tpu_variables_assign[key] for key in name_value_dict.keys()]
        self._tpu_session.run(operations, feed_dict)

    def get_variables_value(self, names_list):
        """
        get the values of variables in name_list:

        Args:
            name_list: a list of variable names. (string)

        Returns:
            A dictionary {variable name: variable value}

        """
        self._sync_down()
        _vars = {name: self._cpu_variables[name] for name in names_list}
        result = self._cpu_session.run(_vars)
        return result

    def import_variables(self, values):
        """
        There are times that you want to import the values of some variables in the beginning of training.
        for instance, you want to import the weights of pre-trained BERT in the beginning of training. in these cases
        you can use set_variables_value method, but you should get the variables name by get_variables_name and identify the
        intended variables. In order to reduce this challenge, this method is defined. You can set the import variables
        as an argument of ModelSpec in model_fn and then import the values by this method.
        Args:
             a list of values (numpy arrays, ...) which corresponds to import_variables in ModelSpec
        """
        if self._cpu_model_spec.import_variables is None:
            raise ValueError("import_variables is not specified in ModelSpec")
        names = [v.name for v in self._cpu_model_spec.import_variables]
        name_value_dict = dict(zip(names, values))
        self.set_variables_value(name_value_dict)

    def export_model(self):
        """
        Returns the value of trainable variables

        Returns:
             a list of values corresponding to trainable_variables in ModelSpec
        """
        if self._cpu_model_spec.import_variables is None:
            raise ValueError("import_variables is not specified in ModelSpec")
        name_list = [v.name for v in self._cpu_model_spec.trainable_variables]
        result = self.get_variables_value(name_list)
        return list(result.values())

    def run(self, rounds):
        """
        Run the training loop for specified number of rounds.
        each round consist of some steps of training and some steps of validation according to run_config
        see RunConfig for more information.

        Args:
            rounds: the Number of rounds
        """
        q = rounds // self._run_config.save_every_rounds
        r = rounds % self._run_config.save_every_rounds
        rounds = q * [self._run_config.save_every_rounds]
        if r > 0:
            rounds.append(r)
        for num_round in rounds:
            feed_dict = {self._num_round: num_round}
            self._cpu_session.run(self._run_opt, feed_dict)
            print("Saving checkpoint ...")
            self._save_checkpoint()
            print("checkpoint saved \n\n")

    def restore(self, checkpoint_path, restore_data_state=False):
        """

        Restore from a checkpoint file. if the model_dir in run_config is not empty, the checkpoint
        wil be restored automatically from the latest checkpoint.
        if restore_data_state is True, then the data checkpoint will also be restored

        """
        print("restoring checkpoint ...")
        with self._cpu_graph.as_default():
            with self._cpu_session.as_default():
                self._checkpoint.restore(checkpoint_path).run_restore_ops()
                if restore_data_state:
                    self._restore_data_op.run()
                else:
                    self._cpu_session.run([self._train_step.initializer, self._dev_step.initializer])
        self._sync_up()
        print("restoring checkpoint completed.\n\n")
