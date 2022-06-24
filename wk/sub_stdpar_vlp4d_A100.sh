#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g jh220031a
#PJM --mpi proc=1

module load nvidia/22.5 nvmpi/22.5

../build/miniapps/vlp4d/stdpar/vlp4d SLD10_large.dat
