#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=1

module purge
module load nvidia/22.5 nvmpi/22.5

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no

#../build/miniapps/vlp4d/openacc/vlp4d TSI20.dat
../build/miniapps/vlp4d/openacc/vlp4d SLD10_large.dat
