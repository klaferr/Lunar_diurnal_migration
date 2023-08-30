#!/bin/bash

#SBATCH -A bramsona
#SBATCH -t 0-06:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

python ../basic_run.py -molecule "H2O" -noon 0 -time 24 -dt 0.25 -num 100 -dir "/home/klaferri/Desktop/Research/Lunar_diurnal_migration/Data/Kris/Results/"

