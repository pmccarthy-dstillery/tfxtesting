# Test pipelines on TFX

## Local
 * depends on the shared `config.yaml` and `pjm_trainer.py`
 * run with `python -m beam_local`

## Cloud
 * partial dependency on `config.yaml` and full dependency on `pjm_trainer.py`
 * to run 
   + update `conf_values.sh`
   + cd into the `kfp_pipeline` dir
   + `source ../conf_values.sh && ../tfx_create.sh`
