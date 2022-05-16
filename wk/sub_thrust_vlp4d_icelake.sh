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

../build/vlp4d/thrust/vlp4d SLD10_large.dat
