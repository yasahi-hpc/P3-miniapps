#!/bin/bash
#PJM -L "node=2"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=16

module load gcc/8.3.1
module load cuda/11.2
module load ompi-cuda/4.1.1-11.2

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no
#export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
#export CUDA_LAUNCH_BLOCKING=1

mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 8 \
    ../build/vlp4d_mpi/kokkos/vlp4d_mpi --num_threads 1 --teams 1 --device 0 --num_gpus 8 --device_map 1 -f SLD10.dat
