#!/bin/bash
#PJM -L "node=2"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=16

module purge
module load nvidia/22.5 nvmpi/22.5

#mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 8 \
#nsys profile ../build/vlp4d_mpi/openacc/vlp4d_mpi --num_threads 1 --teams 1 --device 0 --num_gpus 8 --device_map 1 -f prof_SLD10.dat

mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 8 \
./ompi_bind.sh ../build/miniapps/vlp4d_mpi/openacc/vlp4d_mpi --num_threads 1 --teams 1 --device 0 --num_gpus 8 --device_map 1 -f prof_SLD10.dat
