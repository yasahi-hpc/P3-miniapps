#!/bin/bash

NGPUS=`nvidia-smi -L | wc -l`
export CUDA_VISIBLE_DEVICES=$((MPT_LRANK % NGPUS))
exec $*
