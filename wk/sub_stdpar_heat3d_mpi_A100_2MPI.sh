#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=2

module purge
module load nvidia/22.5
module use /work/opt/local/x86_64/cores/nvidia/22.5/Linux_x86_64/22.5/comm_libs/hpcx/latest/modulefiles
module load hpcx-ompi
module list

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no

mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 2 \
    ./wrapper.sh ../build/miniapps/heat3d_mpi/stdpar/heat3d_mpi --px 1 --py 1 --pz 2 --nx 512 --ny 512 --nz 256 --nbiter 1000 --freq_diag 0
