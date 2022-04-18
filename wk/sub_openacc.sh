#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=1

module load nvidia/22.2 nvmpi/22.2 cmake

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no

../build/vlp4d/openacc/vlp4d SLD10_large.dat
#../build/src/openacc/heat3D
