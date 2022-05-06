#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=2

module purge
module load intel/2021.2.0 impi/2021.2.0 fftw/3.3.9

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no

export OMP_NUM_THREADS=36
export OMP_PROC_BIND=true

mpiexec.hydra -n 2 \
    ../build/heat3d_mpi/openmp/heat3d_mpi --px 2 --py 1 --pz 1 --nx 256 --ny 512 --nz 512 --nbiter 1000
