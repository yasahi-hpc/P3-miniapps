#!/bin/sh

NGPUS=`nvidia-smi -L | wc -l`
export CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_LOCAL_RANK % NGPUS))
exec $*
