#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=8

module purge
module load nvidia/22.5 nvmpi/22.5

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no

#mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 8 -mca use_eager_rdma 1 \
#mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 8 -mca btl_openib_want_cuda_gdr 1 \
mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 8 -mca mpi_common_cuda_gpu_mem_check_workaround 1 \
    ./wrapper.sh ../build/heat3d_mpi/stdpar/heat3d_mpi --px 2 --py 2 --pz 2 --nx 256 --ny 256 --nz 256 --nbiter 1000 --freq_diag 0
