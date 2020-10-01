# source conf_values.sh

tfx pipeline create \
    --pipeline-path=kubeflow_dag_runner.py \
    --endpoint=$ENDPOINT \
    --build-target-image=$CUSTOM_TFX_IMAGE