#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=1

module purge
module load gcc/8.3.1
module load cuda/11.2
module load ompi-cuda/4.1.1-11.2

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no

export OMP_NUM_THREADS=36
export OMP_PROC_BIND=true

mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 1 \
    ../build/heat3d_mpi/thrust/heat3d_mpi --px 1 --py 1 --pz 1 --nx 512 --ny 512 --nz 512 --nbiter 1000 --freq_diag 0
#mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 1 \
#    ../build/heat3d_mpi/thrust/heat3d_mpi --px 1 --py 1 --pz 1 --nx 512 --ny 512 --nz 512 --nbiter 1000 --freq_diag 0
