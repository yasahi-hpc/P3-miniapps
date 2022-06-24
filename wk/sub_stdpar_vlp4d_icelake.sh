#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=1

module purge
module load nvidia/22.5 nvmpi/22.5 fftw/3.3.9

export OMP_NUM_THREADS=36
export OMP_PROC_BIND=true

../build/miniapps/vlp4d/stdpar/vlp4d SLD10_large.dat
