#!/bin/bash

#SBATCH -A bramsona
#SBATCH -t 0-10:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

python ../create_smooth.py -molecule "H2O" -noon 0 -num 10000 -dirc "/home/klaferri/Desktop/Research/Lunar_diurnal_migration/Data/Kris/Results/"


for i in $(seq 10)
do
   srun --cores=3 --mem=5g python ../smooth_parallel_run.py -molecule "H2O" -noon 0 -time 24 -dt 0.25 -num 10000 -segment $i -dirc "/home/klaferri/Desktop/Research/Lunar_diurnal_migration/Data/Kris/Results/"
done



