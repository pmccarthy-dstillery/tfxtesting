#!/bin/bash
set -x

source conf_values.sh

gcloud container clusters get-credentials $CLUSTER_NAME --zone=us-central1-a
export ENDPOINT=$(kubectl describe configmap inverse-proxy-config | grep googleusercontent)

pushd $HOME


export PATH=$HOME:$PATH

popd;

set +x
