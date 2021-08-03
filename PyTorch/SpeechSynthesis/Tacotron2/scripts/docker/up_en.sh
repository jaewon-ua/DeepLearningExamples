#!/bin/bash

set -euxo pipefail

OPTARGS=""
if [[ $(hostname) == "nipa2020-0909" ]]; then
  echo "running on $(hostname)"
  OPTARGS="-e OPENBLAS_CORETYPE=nehalem"
fi

dataroot=/mnt/data
tmproot=/mnt/tmp
CONTANER_NAME=tacotron2
docker stop $CONTANER_NAME || true
docker run --gpus 1 --shm-size=1g -d --ulimit memlock=-1 --ulimit stack=67108864 --rm \
  --network=nlpdemo_default --network-alias=tts --name=tacotron2 \
  --ipc=host -v $PWD:/workspace/tacotron2 \
  -v ${tmproot}:/mnt/tmp -v ${dataroot}:/mnt/data \
  $OPTARGS $CONTANER_NAME
