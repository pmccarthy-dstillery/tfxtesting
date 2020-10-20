#!/bin/bash
export CLUSTER_NAME=cluster-5
export GOOGLE_CLOUD_PROJECT=dst-mlpipes
export ZONE=us-central1-a
export RUN_DATETIME=$(date '+%Y%m%d%H%M%S')
export ENDPOINT=$(kubectl describe configmap inverse-proxy-config | grep googleusercontent)
