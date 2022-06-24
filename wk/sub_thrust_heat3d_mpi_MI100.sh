#!/bin/bash
#SBATCH -J poi_adam
#SBATCH -p amdrome
#SBATCH -w amd1
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=1
#SBATCH --time 00:10:00
#SBATCH -o ./stdout_%J
#SBATCH -e ./stderr_%J

module purge
module load openmpi/4.1.1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
ROCR_VISIBLE_DEVICES=1,2,3 mpirun -n ${SLURM_NTASKS} ../build/miniapps/heat3d_mpi/thrust/heat3d_mpi --nx 512 --ny 512 --nz 256 --px 1 --py 1 --pz 2 --nbiter 1000 --freq_diag 0
