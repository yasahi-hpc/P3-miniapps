#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=1

module load nvidia/21.3

export OMP_NUM_THREADS=36
export OMP_PROC_BIND=true
../build/heat3D/stdpar/heat3D
