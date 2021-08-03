#!/bin/bash

set -euxo pipefail

OPTARGS=""
if [[ $(hostname) == "nipa2020-0909" ]]; then
  echo "running on $(hostname)"
  OPTARGS="-e OPENBLAS_CORETYPE=nehalem"
fi

CONTAINER=tacotron_kr
docker stop $CONTAINER || true

dataroot=/mnt/data
tmproot=/mnt/tmp
export TACOTRON_PATH=/mnt/data/pretrained/kss/checkpoint_Tacotron2_6000
docker run --gpus 1 --shm-size=1g -d --ulimit memlock=-1 --ulimit stack=67108864 --rm \
  --network=nlpdemo_default --network-alias=tts_kr --name=tacotron_kr \
  --ipc=host -v $PWD:/workspace/tacotron2 \
  -v ${tmproot}:/mnt/tmp -v ${dataroot}:/mnt/data \
  -e TACOTRON_PATH \
  $OPTARGS tacotron2
