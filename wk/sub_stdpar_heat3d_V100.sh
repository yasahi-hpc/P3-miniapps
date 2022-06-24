#!/bin/bash
#PBS -q sg8
#PBS -l select=1:ncpus=12:mpiprocs=1:ompthreads=12:ngpus=1
#PBS -l walltime=01:00:00
#PBS -P CityLBM@PG22010

cd $PBS_O_WORKDIR
. /etc/profile.d/modules.sh

module purge
module load cuda/11.0 gnu/7.4.0 nvidia/22.3 openmpi-gdr/4.1.4

../build/miniapps/heat3d/stdpar/heat3d --nx 512 --ny 512 --nz 512 --nbiter 1000 --freq_diag 0
