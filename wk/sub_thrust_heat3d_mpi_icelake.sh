#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=8

module purge
module load gcc/8.3.1
module load cuda/11.2
module load ompi-cuda/4.1.1-11.2

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no

export OMP_NUM_THREADS=9
export OMP_PROC_BIND=true

mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 8 \
    ../build/heat3d_mpi/thrust/heat3d_mpi --px 2 --py 2 --pz 2 --nx 256 --ny 256 --nz 256 --nbiter 1000
