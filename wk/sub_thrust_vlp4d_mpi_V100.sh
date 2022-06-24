#!/bin/bash
#PBS -q sg8
#PBS -l select=1:ncpus=24:mpiprocs=2:ompthreads=12:ngpus=2
#PBS -l walltime=01:00:00
#PBS -P CityLBM@PG22010

cd $PBS_O_WORKDIR
. /etc/profile.d/modules.sh

module purge
module load cuda/11.0 gnu/7.4.0 mpt/2.23-ga

export MPI_SHEPHERD=1
export MPI_USE_CUDA=1

mpirun -np 2 ../build/miniapps/vlp4d_mpi/thrust/vlp4d_mpi --num_threads 1 --teams 1 --device 0 --num_gpus 4 --device_map 1 -f SLD10.dat
