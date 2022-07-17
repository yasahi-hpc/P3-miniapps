#!/bin/bash
#PBS -q sg8
#PBS -l select=1:ncpus=24:mpiprocs=2:ompthreads=12:ngpus=2
#PBS -l walltime=01:00:00
#PBS -P CityLBM@PG22010

cd $PBS_O_WORKDIR
. /etc/profile.d/modules.sh

module purge
module load cuda/11.0 gnu/7.4.0 mpt/2.23-ga

mpirun -np 2 ../build/miniapps/heat3d_mpi/thrust/heat3d_mpi --px 1 --py 1 --pz 2 --nx 512 --ny 512 --nz 256 --nbiter 1000 --freq_diag 0
