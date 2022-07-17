#!/bin/bash
#SBATCH -J poi_adam
#SBATCH -p amdrome
#SBATCH -w amd1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=1
#SBATCH --time 00:10:00
#SBATCH -o ./stdout_%J
#SBATCH -e ./stderr_%J

module purge
module load openmpi/4.1.1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
ROCR_VISIBLE_DEVICES=1,2,3 mpirun -n ${SLURM_NTASKS} ../build/miniapps/heat3d_mpi/kokkos/heat3d_mpi --nx 512 --ny 512 --nz 512 --px 1 --py 1 --pz 1 --nbiter 1000 --freq_diag 0 --num_threads 1 --teams 1 --device 0 --num_gpus 3 --device_map 1
