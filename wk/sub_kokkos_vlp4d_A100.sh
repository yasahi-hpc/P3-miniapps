#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=1

module load gcc/8.3.1
module load cuda/11.2
module load ompi-cuda/4.1.1-11.2

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no

../build/miniapps/vlp4d/kokkos/vlp4d SLD10_large.dat
#../build/vlp4d/kokkos/vlp4d TSI20.dat
