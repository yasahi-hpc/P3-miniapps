#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=1
###PJM --omp thread=36

module purge
module load nvidia/22.5 nvmpi/22.5

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no

export OMP_NUM_THREADS=36
export OMP_PROC_BIND=true

#mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC --map-by ppr:1:socket:PE=36 \
    #    ../build/miniapps/heat3d_mpi/stdpar/heat3d_mpi --px 1 --py 1 --pz 1 --nx 512 --ny 512 --nz 512 --nbiter 1000 --freq_diag 0
mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -report-bindings --map-by ppr:1:socket:PE=36 \
        ../build/miniapps/heat3d_mpi/stdpar/heat3d_mpi --px 1 --py 1 --pz 2 --nx 512 --ny 512 --nz 256 --nbiter 1000 --freq_diag 0
