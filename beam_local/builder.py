import datetime
import logging

from beam_local import config

import tensorflow_model_analysis as tfma
from tfx.components import (
    Evaluator,
    ExampleValidator,
    ImportExampleGen,
    Pusher,
    ResolverNode,
    SchemaGen,
    StatisticsGen,
    Trainer
)
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import (
    metadata,
    pipeline
)
from tfx.proto import (pusher_pb2,trainer_pb2)
from tfx.types import Channel
from tfx.types.standard_artifacts import (Model, ModelBlessing)

conf = config.load()

def build_pipeline():
    """
    Gather tfx components and produce the output pipeline
    """

    conf['serving_model_dir'] = f"{conf['serving_model_dir']}/OL{653374}/{datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d%H%m')}"
    logging.info("Serving model dir is now %s",conf['serving_model_dir'])

    example_gen = ImportExampleGen(input_base=conf['train_data'])

    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=False
    )
    
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    trainer = Trainer(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=conf['trainer_module_file'],
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor), # define this to use run_fn instead of trainer_fn
        train_args=trainer_pb2.TrainArgs(num_steps=conf['train_args_steps']),
        eval_args=trainer_pb2.EvalArgs(num_steps=50)
    )


    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(label_key='label')
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    # tfma.MetricConfig(class_name='ExampleCount'),
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
            tfma.SlicingSpec(),
        ])

    model_resolver = ResolverNode(
          instance_name='latest_blessed_model_resolver',
          resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
          model=Channel(type=Model),
          model_blessing=Channel(type=ModelBlessing))

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)

    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=conf['serving_model_dir'])))

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        trainer,
        model_resolver,
        evaluator,
        pusher
    ]

    tfx_pipeline = pipeline.Pipeline(
        pipeline_name=conf['pipeline_name'],
        pipeline_root=conf['pipeline_root_dir'],
        components=components,
        enable_cache=False,
        metadata_connection_config=(
            metadata.sqlite_metadata_connection_config(conf['metadata_path'])

        )
    )

    return tfx_pipeline