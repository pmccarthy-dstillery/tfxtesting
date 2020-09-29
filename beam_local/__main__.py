import datetime
import logging
import shutil

from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

from beam_local import config, builder

conf = config.load()

timestamp = datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d%H%m')


if __name__ == "__main__":

    # try:
    # shutil.rmtree(conf['metadata_path'])
    # except FileNotFoundError as e:
        # pass

    logging.basicConfig(level='INFO')

    tfx_pipeline = builder.build_pipeline(timestamp)

    BeamDagRunner().run(tfx_pipeline)
