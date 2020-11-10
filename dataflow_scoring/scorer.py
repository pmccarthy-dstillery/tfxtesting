import apache_beam as beam
from apache_beam.options.pipeline_options import (PipelineOptions, SetupOptions) 
import argparse

OUTPUT_SCHEMA = "bestdeviceid:STRING, bestdeviceidtype:STRING, offerlevel:INTEGER, score:FLOAT, timestamp:TIMESTAMP"    

OFFERLEVELS = [645570,640602,652131,640737,643902,649216,31454,633602,622270,636785]

def map_to_dict(x):
    ""
    return {
        'bestdeviceid':x['bestdeviceid'], 
        'bestdeviceidtype':x['bestdeviceidtype'],
        'offerlevel': x['offerlevel'],
        'score':x['score'],
        'timestamp':x['timestamp']}


class ModelInference(beam.DoFn):

    def __init__(self, offerlevel_id):
        
        import datetime
        from google.cloud import storage
        import joblib
        
        self._dt = datetime
        
        self._offerlevel = offerlevel_id
        self._bucket = 'pjm-sklearn-models'
        self._filename = f"ol{offerlevel_id}.joblib"
        
        _bucket = storage.Client().get_bucket(self._bucket)
        _blob = _bucket.blob(self._filename)
        
        # download to local
        _blob.download_to_filename(self._filename)
        self.model = joblib.load(self._filename)


    def process(self, element):
        
        pred = self.model.predict_proba([element['visitdata']])[:,1]
        
        return [{
            'bestdeviceid':element['bestdeviceid'],
            'bestdeviceidtype':element['bestdeviceidtype'],
            'offerlevel':int(self._offerlevel),
            'score':pred[0],
            'timestamp':self._dt.datetime.now().isoformat()
        }]


def run(argv=None, save_main_session=True):

    parser = argparse.ArgumentParser()
    known_args, pipeline_args = parser.parse_known_args(argv)

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

    with beam.Pipeline(options=pipeline_options) as p:
        source_data = (p 
        | 'QueryTable' >> beam.io.Read(
            beam.io.BigQuerySource(
                query="""
                    select *
                    from [dst-mlpipes:pjm_visitdata_sample.universe]
                    limit 1000000
                """)
            )
        )

        inferences = []
        for offerlevel in OFFERLEVELS:
            inferences.append(source_data 
                | f'Perform Inference OL {offerlevel}' >> beam.ParDo(ModelInference(offerlevel)))

        outputs = (
            tuple(inferences)
        | 'Combine outputs' >> beam.Flatten()
        # | 'Map to necessary structure' >> beam.Map(map_to_dict)
        | 'Write' >> beam.io.WriteToBigQuery(
            table='scoring_output',
            dataset='pjm_visitdata_sample',
            project='dst-mlpipes',
            schema=OUTPUT_SCHEMA)
        )

if __name__ == '__main__':
    run()