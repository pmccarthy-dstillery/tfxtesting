source ../conf_values.sh

SKAFFOLD=./skaffold
KUBE_CONFIG=$HOME/.kube/config

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

tfx pipeline update \
    --pipeline-path=kubeflow_dag_runner.py \
    --endpoint=$ENDPOINT 