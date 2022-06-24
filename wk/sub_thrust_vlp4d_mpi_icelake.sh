#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=30:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=2

module purge
module load gcc/8.3.1
module load cuda/11.2
module load ompi-cuda/4.1.1-11.2

export OMP_NUM_THREADS=36
export OMP_PROC_BIND=true

mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 2 \
    ../build/miniapps/vlp4d_mpi/thrust/vlp4d_mpi --num_threads 36 --teams 1 --device 0 --num_gpus 8 --device_map 1 -f SLD10.dat
