#!/bin/bash
#SBATCH --job-name=P16_M262k
#SBATCH -N 2
#SBATCH --ntasks-per-node=16
#SBATCH --output=P16_M262k_%j.out
#SBATCH --error=P16_M262k_%j.err
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
MSIZES=(262144)
PSIZES=(16)


#Automated Loop
    for P in "${PSIZES[@]}"; do
    for M in "${MSIZES[@]}"; do
        for i in {1..5}; do
             mpirun -np $P ./src $M $D1 $D2 $T $SEED
        done
        echo ""
    done
done

