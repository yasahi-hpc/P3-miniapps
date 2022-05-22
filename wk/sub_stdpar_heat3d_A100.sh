#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=30:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=1

module purge
module load nvidia/22.2 nvmpi/22.2
export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no

../build/heat3d/stdpar/heat3d --nx 512 --ny 512 --nz 512 --nbiter 50000 --freq_diag 1000
