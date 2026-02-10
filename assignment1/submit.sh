#!/bin/bash
#SBATCH --job-name=mpi_dist_exchange
#SBATCH -N 2
#SBATCH --ntasks-per-node=16
#SBATCH --output=exchange_%j.out
#SBATCH --error=exchange_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:10:00

module purge
module load compiler/oneapi-2024
module load compiler/oneapi-2024/mpi


#parameters
D1=2
D2=4
T=5
SEED=1000
MSIZES=(262144 1048576)
P_COUNTS=(8)


#Automated Loop
for P in "${P_COUNTS[@]}"; do
    for M in "${MSIZES[@]}"; do
        for i in {1..5}; do
             mpirun -np $P ./seqDivakar $M $D1 $D2 $T $SEED
        done
    done
done
