#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=2

module purge
module load intel/2021.2.0 impi/2021.2.0 fftw/3.3.9

export OMP_NUM_THREADS=36
export OMP_PROC_BIND=true

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no

mpiexec.hydra -n 2 \
    ../build/vlp4d_mpi/openmp/vlp4d_mpi --num_threads 36 --teams 1 --device 0 --num_gpus 8 --device_map 1 -f SLD10.dat
