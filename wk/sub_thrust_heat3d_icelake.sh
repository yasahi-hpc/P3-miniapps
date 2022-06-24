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

export OMP_NUM_THREADS=36
export OMP_PROC_BIND=true

../build/miniapps/heat3d/thrust/heat3d --nx 512 --ny 512 --nz 512 --nbiter 1000 --freq_diag 0
