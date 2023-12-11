#!/bin/bash

#SBATCH -A bramsona
#SBATCH -t 0-20:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --ntasks=10

python ../create_smooth.py -molecule "H2O" -noon 0 -num 100 -dirc "/home/klaferri/Desktop/Research/Lunar_diurnal_migration/Data/Kris/Results/"

for jjj in $(seq 10)
do
   srun --exclusive -N1 --ntasks=1 --mem-per-cpu=4G python ../rough_parallel_run.py -molecule "H2O" -scale 57 -noon 0 -time 24 -dt 0.25 -num 100 -segment $jjj -dirc "/home/klaferri/Desktop/Research/Lunar_diurnal_migration/Data/Kris/Results/" &
done

wait

