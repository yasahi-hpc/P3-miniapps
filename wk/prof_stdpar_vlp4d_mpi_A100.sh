#!/bin/bash
#PJM -L "node=2"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=16

module purge
module load nvidia/22.5
module use /work/opt/local/x86_64/cores/nvidia/22.5/Linux_x86_64/22.5/comm_libs/hpcx/latest/modulefiles
module load hpcx-ompi
module list

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=no
export UCX_RNDV_FRAG_MEM_TYPE=cuda

mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 8 \
./ompi_bind.sh ../build/miniapps/vlp4d_mpi/stdpar/vlp4d_mpi --num_threads 1 --teams 1 --device 0 --num_gpus 8 --device_map 1 -f prof_SLD10.dat
