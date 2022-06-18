#!/bin/sh

NGPUS=3
#NGPUS=`nvidia-smi -L | wc -l`
export ROCR_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_LOCAL_RANK % NGPUS + 1))
exec $*
