#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=1

module purge
module load intel/2021.2.0 impi/2021.2.0 fftw/3.3.9

export OMP_NUM_THREADS=36
export OMP_PROC_BIND=true

../build/vlp4d/openmp/vlp4d SLD10_large.dat
