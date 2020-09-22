import absl
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

import os
import pprint
import tempfile
import urllib

import absl
import tensorflow as tf
import tensorflow_model_analysis as tfma
tf.get_logger().propagate = False
pp = pprint.PrettyPrinter()

import tfx
from tfx.components import CsvExampleGen, ImportExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.utils.dsl_utils import external_input



print('TensorFlow version: {}'.format(tf.__version__))
print('TFX version: {}'.format(tfx.__version__))

_tfx_root='/home/pmccarthy/meta/pipeline0824'
_pjm_root = os.path.join(_tfx_root, 'pjm_pipeline')

_serving_model_dir=os.path.join(tempfile.mkdtemp(), 'serving_model/pjm_pipeline')
absl.logging.set_verbosity(absl.logging.INFO)

_data_root='gs://pjm-predict-bucket'

# context = InteractiveContext()

example_gen = ImportExampleGen(input=external_input(_data_root))

# context.run(example_gen)

statistics_gen = StatisticsGen(
    examples=example_gen.outputs['examples']
)

# context.run(statistics_gen)

schema_gen = SchemaGen(
    statistics=statistics_gen.outputs['statistics'],
    infer_feature_shape=False
)
# context.run(schema_gen)

example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema']
)
# context.run(example_validator)

_pjm_trainer_module_file = 'pjm_trainer.py'

trainer = Trainer(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=_pjm_trainer_module_file,
    custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor), # define this to use run_fn instead of trainer_fn
    train_args=trainer_pb2.TrainArgs(num_steps=1024),
    eval_args=trainer_pb2.EvalArgs(num_steps=50)
)

# context.run(trainer)

eval_config = tfma.EvalConfig(
    model_specs=[
        # This assumes a serving model with signature 'serving_default'. If
        # using estimator based EvalSavedModel, add signature_name: 'eval' and 
        # remove the label_key.
        tfma.ModelSpec(label_key='label')
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            # The metrics added here are in addition to those saved with the
            # model (assuming either a keras model or EvalSavedModel is used).
            # Any metrics added into the saved model (for example using
            # model.compile(..., metrics=[...]), etc) will be computed
            # automatically.
            # To add validation thresholds for metrics saved with the model,
            # add them keyed by metric name to the thresholds map.
            metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(class_name='BinaryAccuracy',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': 0.5}),
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-10})))
            ]
        )
    ],
    slicing_specs=[
        # An empty slice spec means the overall slice, i.e. the whole dataset.
        tfma.SlicingSpec(),
        # Data can be sliced along a feature column. In this case, data is
        # sliced along feature column trip_start_hour.
#         tfma.SlicingSpec(feature_keys=['trip_start_hour'])
    ])

model_resolver = ResolverNode(
      instance_name='latest_blessed_model_resolver',
      resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(type=ModelBlessing))
# context.run(model_resolver)

evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    # Change threshold will be ignored if there is no baseline (first run).
    eval_config=eval_config)
# context.run(evaluator)

pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=_serving_model_dir)))
# context.run(pusher)

_runner_type = 'beam' 
_pipeline_name = 'pjm_%s' % _runner_type



# _notebook_filepath = os.path.join(os.getcwd(),
#                                   'InteractiveCA.ipynb')

# TODO(USER): Fill out the paths for the exported pipeline.
# _tfx_root = os.path.join(os.environ['HOME'], 'tfx')
# _taxi_root = os.path.join(os.environ['HOME'], 'taxi')
# _serving_model_dir = os.path.join(_pjm_root, 'serving_model')
# _data_root = os.path.join(_taxi_root, 'data', 'simple')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')

components = [
    example_gen, statistics_gen, schema_gen, example_validator, #transform,
    trainer, model_resolver, evaluator, pusher
]



absl.logging.set_verbosity(absl.logging.INFO)

tfx_pipeline = pipeline.Pipeline(
    pipeline_name=_pipeline_name,
    pipeline_root=_pipeline_root,
    components=components,
    enable_cache=True,
    metadata_connection_config=(
        metadata.sqlite_metadata_connection_config(_metadata_path)),

    # We use `--direct_num_workers=1` by default to launch 1 Beam worker process
    # during Beam DirectRunner component execution. This mitigates issues with
    # GPU memory usage when many workers are run sharing GPU resources.  Change
    # this to `--direct_num_workers=0` to run one worker per available CPU
    # thread or `--direct_num_workers=$N`, where `$N` is a fixed number of
    # worker processes.
    #
    # TODO(b/142684737): The Beam multi-processing API might change.
    beam_pipeline_args = ['--direct_num_workers=1'],

    additional_pipeline_args={})

BeamDagRunner().run(tfx_pipeline)
