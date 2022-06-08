#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=1

module purge
module load intel impi/2021.2.0 fftw/3.3.9

export OMP_NUM_THREADS=36
export OMP_PROC_BIND=true

mpiexec.hydra -n 1 \
    ../build/heat3d_mpi/kokkos/heat3d_mpi --px 1 --py 1 --pz 1 --nx 512 --ny 512 --nz 512 --nbiter 1000 --freq_diag 0 --num_threads 36 --teams 1 --device 0 --num_gpus 8 --device_map 1
#mpiexec.hydra -n 2 \
#    ../build/heat3d_mpi/kokkos/heat3d_mpi --px 1 --py 1 --pz 2 --nx 512 --ny 512 --nz 256 --nbiter 1000 --freq_diag 0 --num_threads 36 --teams 1 --device 0 --num_gpus 8 --device_map 1
