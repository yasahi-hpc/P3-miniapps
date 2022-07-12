#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=2

module purge
module load nvidia/22.5 nvmpi/22.5 fftw/3.3.9

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no

export OMP_NUM_THREADS=36
export OMP_PROC_BIND=true

mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC --map-by ppr:1:socket:PE=36 \
    ../build/miniapps/vlp4d_mpi/stdpar/vlp4d_mpi --num_threads 36 --teams 1 --device 0 --num_gpus 0 --device_map 1 -f SLD10.dat
