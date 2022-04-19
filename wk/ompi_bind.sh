#!/bin/bash
gr=${OMPI_COMM_WORLD_RANK:-0}
lr=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
# GPU affinity
NUM_GPUS=${NUM_GPUS:-8}
gpuid=`expr ${lr} \% ${NUM_GPUS}`
export CUDA_VISIBLE_DEVICES=${gpuid}
nsys profile $@
