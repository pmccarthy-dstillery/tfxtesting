#!/bin/bash
set +x

source conf_values.sh

gcloud container clusters get-credentials $CLUSTER_NAME --zone=us-central1-a

pushd $HOME

curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/latest/skaffold-linux-amd64 
chmod +x skaffold

pip install -U pip --no-input
pip install tfx kfp --use-feature=2020-resolver 

export GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT

popd;

set -x
