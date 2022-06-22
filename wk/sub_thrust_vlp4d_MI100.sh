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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
ROCR_VISIBLE_DEVICES=1,2,3 srun ../build/vlp4d/thrust/vlp4d SLD10_large.dat
