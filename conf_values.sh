#!/bin/bash
set -x
export CLUSTER_NAME=cluster-5
export GOOGLE_CLOUD_PROJECT=dst-mlpipes
export ZONE=us-central1-a
export PIPELINE_NAME=pjm-pipeline-1001
export RUN_DATETIME=$(date '+%Y%m%d%H%M%S')
export ENDPOINT=$(kubectl describe configmap inverse-proxy-config | grep googleusercontent)
export CUSTOM_TFX_IMAGE="gcr.io/${GOOGLE_CLOUD_PROJECT}/${PIPELINE_NAME}"
export GCS_BUCKET_NAME=pjm-kfp-pipelines/${PIPELINE_NAME}/${RUN_DATETIME}
set +x
