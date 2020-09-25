import logging

from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

from beam_local import config, builder

conf = config.load()


if __name__ == "__main__":

    logging.basicConfig(level='INFO')

    tfx_pipeline = builder.build_pipeline()

    BeamDagRunner().run(tfx_pipeline)