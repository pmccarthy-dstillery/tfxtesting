#!/bin/bash
set -x
export CLUSTER_NAME=cluster-5
export GOOGLE_CLOUD_PROJECT=dst-mlpipes
export ZONE=us-central1-a
export PIPELINE_NAME=pjm-pipeline
export ENDPOINT=$(kubectl describe configmap inverse-proxy-config | grep googleusercontent)
export CUSTOM_TFX_IMAGE="gcr.io/${GOOGLE_CLOUD_PROJECT}/tfx-pipeline-20200930"
export GCS_BUCKET_NAME=pjm-kfp-pipelines/${PIPELINE_NAME}/$(date '+%Y%m%d%H%M%S')
set +x
