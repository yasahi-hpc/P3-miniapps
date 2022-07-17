#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=2

module purge
module load gcc/8.3.1
module load cuda/11.2
module load ompi-cuda/4.1.1-11.2
#module load cuda/11.4
#module load ompi-cuda/4.1.1-11.4

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no

mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 2 \
    ../build/miniapps/heat3d_mpi/kokkos/heat3d_mpi --px 1 --py 1 --pz 2 --nx 512 --ny 512 --nz 256 --nbiter 1000 --freq_diag 0 --num_threads 1 --teams 1 --device 0 --num_gpus 8 --device_map 1
#mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 8 \
#    ../build/miniapps/heat3d_mpi/kokkos/heat3d_mpi --px 2 --py 2 --pz 2 --nx 256 --ny 256 --nz 256 --nbiter 1000 --num_threads 1 --teams 1 --device 0 --num_gpus 8 --device_map 1
