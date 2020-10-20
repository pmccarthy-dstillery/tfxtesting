#!/bin/bash
set -x
source conf_values.sh

SKAFFOLD=./skaffold
KUBE_CONFIG=$HOME/.kube/config
PIPELINE_NAME=$(grep pipeline_name config.yaml | tail -1 | awk '{print tolower($2)}' | sed s/\'//g)

cp pjm_trainer.py kfp_pipeline/models/keras/

if [ -f build.yaml ]; then
    rm build.yaml 
fi

if [ -f Dockerfile ]; then
    rm Dockerfile 
fi

if [ -f "${PIPELINE_NAME}.tar.gz" ]; then
    rm "${PIPELINE_NAME}.tar.gz"
fi

if [ -f "$SKAFFOLD" ]; then
    rm $SKAFFOLD
fi
curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/latest/skaffold-linux-amd64 && chmod +x skaffold
export PATH=$(pwd):$PATH

if [ -f "$KUBE_CONFIG" ]; then
    echo "kube config exists"
else
    gcloud container clusters get-credentials $CLUSTER_NAME --region=$ZONE
fi

pushd kfp_pipeline

if [ -f "build.yaml" ]; then
    rm build.yaml
fi

if [ -f "Dockerfile" ]; then
    rm Dockerfile 
fi

CUSTOM_TFX_IMAGE="gcr.io/${GOOGLE_CLOUD_PROJECT}/${PIPELINE_NAME}"
tfx pipeline create \
    --pipeline-path=kubeflow_dag_runner.py \
    --endpoint=$ENDPOINT \
    --build-target-image=$CUSTOM_TFX_IMAGE
popd
set +x