#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=1

module purge
module load nvidia/22.5 nvmpi/22.5

export OMP_NUM_THREADS=36
export OMP_PROC_BIND=true

../build/miniapps/heat3d/stdpar/heat3d --nx 512 --ny 512 --nz 512 --nbiter 1000 --freq_diag 0
