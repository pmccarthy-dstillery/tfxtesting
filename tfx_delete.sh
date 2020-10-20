set -x
pushd kfp_pipeline
source ../conf_values.sh

SKAFFOLD=./skaffold
KUBE_CONFIG=$HOME/.kube/config
PIPELINE_NAME=$(grep pipeline_name ../config.yaml | tail -1 | awk '{print $2}' | sed s/\'//g)

if [ -f "$KUBE_CONFIG" ]; then
    echo "kube config exists"
else
    gcloud container clusters get-credentials $CLUSTER_NAME --region=$ZONE
fi


tfx pipeline delete \
    --engine=kubeflow \
    --pipeline-name=$PIPELINE_NAME \
    --endpoint=$ENDPOINT 


if [ -d $HOME/kubeflow ]; then
    rm -rf $HOME/kubeflow
fi

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

popd
set +x