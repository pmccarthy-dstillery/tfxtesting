
from tfx.components.trainer.executor import TrainerFnArgs

import tensorflow as tf

import glob
import os


MAX_IDX = 45431

FEATURE_SPEC = {'sparse': tf.io.SparseFeature(index_key='indices',
                                              value_key='values',
                                              dtype=tf.int64,
                                              size=MAX_IDX),
                'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)}


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(
      filenames,
      compression_type='GZIP')


def _input_fn(file_pattern, batch_size):
    print(file_pattern)
  
    def my_parser(x):
        example = tf.io.parse_example(x, FEATURE_SPEC)
        
        return example['sparse'], example['label']

    dataset = (
        tf.data.TFRecordDataset(glob.glob(file_pattern[0]), compression_type='GZIP')
        .batch(batch_size)
        .map(my_parser)
        .repeat()
    )
    
    return dataset


def _build_keras():    

    FULL_DIM = MAX_IDX
    input_layer = tf.keras.layers.Input(FULL_DIM, sparse=True, name='sparse')

    lin_fn = tf.keras.layers.Dense(1, 
                   activation='sigmoid', 
                   kernel_regularizer=tf.keras.regularizers.l2(0.1))(input_layer)
    
    reg_model = tf.keras.Model(inputs = input_layer,
                               outputs = lin_fn)
    reg_model.compile()
    reg_model.summary()    
    
    return reg_model


def run_fn(fn_args: TrainerFnArgs):

    BATCH_SIZE = 1024
    
    train_dataset = _input_fn(fn_args.train_files, BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, BATCH_SIZE)

    model = _build_keras()
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir, update_freq='batch')
    model.fit(train_dataset,
             steps_per_epoch=fn_args.train_steps,
             validation_data=eval_dataset,
             validation_steps=fn_args.eval_steps,
             callbacks=[tensorboard_callback])
        
    signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    None).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)


def _get_serve_tf_examples_fn(model, tf_transform_output=None):
  """Returns a function that parses a serialized tf.Example and applies TFT."""

#   model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = FEATURE_SPEC
    feature_spec.pop('label')
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    
    return model(parsed_features)

  return serve_tf_examples_fn   
