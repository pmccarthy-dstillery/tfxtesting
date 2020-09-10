#!/bin/bash
set -x
curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/latest/skaffold-linux-amd64 
chmod +x skaffold

pip install -U pip --no-input
pip install tfx kfp --use-feature=2020-resolver 
set +x
