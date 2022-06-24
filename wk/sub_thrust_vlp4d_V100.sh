#!/bin/bash
#PBS -q sg8
#PBS -l select=1:ncpus=12:mpiprocs=1:ompthreads=12:ngpus=1
#PBS -l walltime=01:00:00
#PBS -P CityLBM@PG22010

cd $PBS_O_WORKDIR

module purge
module load cuda/11.0 gnu/7.4.0

../build/miniapps/vlp4d/thrust/vlp4d SLD10_large.dat
