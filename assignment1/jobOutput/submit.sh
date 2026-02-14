#!/bin/bash
#SBATCH --job-name=job633
#SBATCH -N 2
#SBATCH --ntasks-per-node=16
#SBATCH --output=job633_%j.out
#SBATCH --error=job633_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:10:00

module purge
module load compiler/oneapi-2024
module load compiler/oneapi-2024/mpi


#parameters
D1=2
D2=4
T=10
SEED=1000
MSIZES=(1048576)
PSIZES=(32)


#Automated Loop
    for P in "${PSIZES[@]}"; do
    for M in "${MSIZES[@]}"; do
        for i in {1..5}; do
             mpirun -np $P ./src $M $D1 $D2 $T $SEED
        done
        echo ""
    done
done

