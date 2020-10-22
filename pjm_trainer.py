
from tfx.components.trainer.executor import TrainerFnArgs

import tensorflow as tf
import tensorflow_transform as tft
import typing

import os


SGD_MAX_IDX = int(50.01e6) 
MAX_IDX = 40*1000

def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(
      filenames,
      compression_type='GZIP')


def _input_fn(file_pattern: typing.List[typing.Text],
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int) -> tf.data.Dataset:

    transformed_feature_spec = (tf_transform_output.transformed_feature_spec().copy())

    dataset = (tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key='label')
        )
    
    return dataset


def preprocessing_fn(inputs):
  """
  Perform feature reduction via `compute_and_apply_vocabulary`. An
  `indices` tensor should come in with values in (0,5e7) and should
  be transformed to (0,40000).
  """

  outputs = {} 

  outputs['label'] = inputs['label']

  outputs['indices'] = (
    tft.compute_and_apply_vocabulary(x=inputs['indices'],
                                     top_k=(MAX_IDX-5),
                                     num_oov_buckets=5,
                                     vocab_filename='my_vocab')
  )
  
  return outputs


class SparseConstructorLayer(tf.keras.layers.Layer):
    
    def __init__(self, n):
        self.n = n
        super(SparseConstructorLayer, self).__init__()
        

    def call(self, inputs):
        row_inds = inputs.indices[:,0]
        col_inds = tf.cast(inputs.values, tf.int64)
        
        indices = tf.transpose(tf.stack([row_inds, col_inds]))
        values = tf.ones(tf.shape(inputs.values))
        dense_shape = [tf.shape(inputs)[0], tf.cast(self.n, tf.int64)]
        
        return tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
        

    def get_config(self):
        return {'n': self.n}


def _build_keras():   

    BATCH_SIZE=1024

    input_layer = tf.keras.layers.Input(shape=MAX_IDX, batch_size=BATCH_SIZE, sparse=True, name='indices')
  
    sparsed_input = SparseConstructorLayer(MAX_IDX)(input_layer)

    lin_fn = tf.keras.layers.Dense(1, 
                   activation='sigmoid', 
                   kernel_regularizer=tf.keras.regularizers.l2())(sparsed_input)
    
    reg_model = tf.keras.Model(inputs = input_layer,
                               outputs = lin_fn)
    reg_model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
      loss='binary_crossentropy',
      metrics=[tf.keras.metrics.AUC()]
    )
    reg_model.summary()    
    
    return reg_model


def run_fn(fn_args: TrainerFnArgs):

    BATCH_SIZE = 1024

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, BATCH_SIZE)
  
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
      model = _build_keras()
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir, update_freq='batch')

    model.fit(train_dataset,
             steps_per_epoch=fn_args.train_steps,
             validation_data=eval_dataset,
             validation_steps=fn_args.eval_steps,
            #  callbacks=[tensorboard_callback])
            callbacks=[])
        
    signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)


def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example and applies TFT."""

  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    # feature_spec = FEATURE_SPEC
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop('label')
    # feature_spec.pop(base.LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    transformed_features = model.tft_layer(parsed_features)
    # transformed_features.pop('label')
    
    return model(transformed_features)

  return serve_tf_examples_fn   
