# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TFX template taxi model.

A DNN keras model which uses features defined in features.py and network
parameters defined in constants.py.
"""

from __future__ import division
from __future__ import print_function

import glob
import os
from absl import logging
import tensorflow as tf
import tensorflow_transform as tft

from models import features
from models.keras import constants


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _get_serve_tf_examples_fn(model, tf_transform_output=None):
  """Returns a function that parses a serialized tf.Example and applies TFT."""

#  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = features.FEATURE_SPEC
    feature_spec.pop(features.LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    return model(parsed_features)

  return serve_tf_examples_fn


def _input_fn(file_pattern, num_steps, batch_size=200):
  """Generates features and label for tuning/training.

  Args:
    file_pattern: input tfrecord file pattern.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """

  def my_parser(x):
    example = tf.io.parse_example(x, features.FEATURE_SPEC)
    
    return example['sparse'], example['label']

#  dataset = (
#        tf.data.TFRecordDataset(glob.glob(file_pattern[0]), compression_type='GZIP')
#        .shuffle(buffer_size=5)
#        .take(num_steps)
#        .batch(batch_size)
#        .map(my_parser)
#    )

  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=features.FEATURE_SPEC,
      reader=_gzip_reader_fn,
      label_key='label')

  return dataset


def _build_keras_model(hidden_units, learning_rate):
  """Creates a DNN Keras model for classifying taxi data.

  Args:
    hidden_units: [int], the layer sizes of the DNN (input layer first).
    learning_rate: [float], learning rate of the Adam optimizer.

  Returns:
    A keras Model.
  """

  input_layer = tf.keras.layers.Input(hidden_units, sparse=True, name='sparse')

  lin_fn = tf.keras.layers.Dense(1, 
                 activation='sigmoid', 
                 kernel_regularizer=tf.keras.regularizers.l2(0.1))(input_layer)
  
  model = tf.keras.Model(inputs = input_layer,
                             outputs = lin_fn)
  model.compile(
      loss='binary_crossentropy',
      optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
      metrics=[tf.keras.metrics.BinaryAccuracy()])
  model.summary(print_fn=logging.info)
  
  return model



# TFX Trainer will call this function.
def run_fn(fn_args):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """

  train_dataset = _input_fn(fn_args.train_files, #fn_args.transform_output,
                            constants.TRAIN_BATCH_SIZE,
                            fn_args.train_steps)
  eval_dataset = _input_fn(fn_args.eval_files, #fn_args.transform_output,
                           constants.EVAL_BATCH_SIZE,
                           fn_args.eval_steps)

  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model(
        hidden_units=constants.HIDDEN_UNITS,
        learning_rate=constants.LEARNING_RATE)
  # This log path might change in the future.
  log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir, update_freq='batch')

  model.fit(
      train_dataset,
      validation_data=eval_dataset,
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
