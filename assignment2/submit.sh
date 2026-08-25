#!/bin/bash
#SBATCH --job-name=mpi_assignment2
#SBATCH -N 2
#SBATCH --ntasks-per-node=48
#SBATCH --output=assign2_%j.out
#SBATCH --error=assign2_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:10:00

module purge
module load compiler/oneapi-2024
module load compiler/oneapi-2024/mpi


D=7
T=5
SEED=1000
F=2
ISO=500
RUNS=5


# Data sizes: 120 and 240
NXYZ=(120 240)

# Process configurations: "P PX PY PZ PPN"
CONFIGS=(
    "32 4 4 2 32"
    "48 6 4 2 48"
    "64 4 4 4 32"
    "96 6 4 4 48"
)



for N in "${NXYZ[@]}"; do
    for CONFIG in "${CONFIGS[@]}"; do
        
        read P PX PY PZ PPN <<< "$CONFIG"

        echo "Configuration: Data Size = ${N}x${N}x${N}"
        echo "Processes (P) = $P | Grid = $PX x $PY x $PZ | PPN = $PPN"

        for i in $(seq 1 $RUNS); do
            echo "--- Run $i ---"
            
            OUTPUT=$(mpirun -np $P ./assignment2 \
                $D $PPN $PX $PY $PZ \
                $N $N $N \
                $T $SEED $F $ISO)
            echo "$OUTPUT"

            TIME=$(echo "$OUTPUT" | tail -n 1)
            echo "$P $N $TIME" >> timing.txt
                
        done
        echo ""
    done
done

