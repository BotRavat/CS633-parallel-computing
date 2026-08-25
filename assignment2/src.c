#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/*
Our Approach: 
1.) Single count of isovalues
2.) Diagonal comparison not considered

We start with laying out single dimentional process ranks in a 3D rubic cube fashion with dimensions px, py, pz. 
Each process is responsible for a local 3D block of size nx*ny*nz.
We identify neighbours based on calculated stencil radius and Array is used for storing neighboring process ranks in each direction (left, right, up, down, front, back).
Derived MPI vector datatype is used for efficient halo exchange in x, y, and z directions.

The main computation loop consists of T iterations.
In each iteration:
1.we first perform halo exchange with neighbors in all 6 directions (if they exist).
2.we compute the new values for the interior cells based on the stencil operation.
3.We count isovalues unidirectionally in x, y, and z directions by detecting sign change for each field and use MPI_Reduce to get the global counts.
4.Buffer swapped for next iteration.

Finally, we print the execution time by root process rank 0.
*/

int main(int argc, char **argv)
{

    // MPI initialization
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Command line argument checking
    if (argc < 13)
    {
        if (rank == 0)
            printf("Input must contain these 12 arguments: d ppn px py pz nx ny nz T seed F isovalue\n");
        MPI_Finalize();
        return 1;
    }

    /*
     * Expected command line arguments:
     * argv[1] = d: stencil size (7 for 7-point stencil)
     * argv[2] = ppn: processes per node
     * argv[3] = px: number of processes in x dimension
     * argv[4] = py: number of processes in y dimension
     * argv[5] = pz: number of processes in z dimension
     * argv[6] = nx: local grid size in x dimension 
     * argv[7] = ny: local grid size in y dimension 
     * argv[8] = nz: local grid size in z dimension
     * argv[9] = T: number of iterations
     * argv[10] = seed: random seed
     * argv[11] = F: number of fields
     * argv[12] = isovalue
     */

    // Parse command line arguments
    int d = atoi(argv[1]);
    int ppn = atoi(argv[2]);
    int px = atoi(argv[3]);
    int py = atoi(argv[4]);
    int pz = atoi(argv[5]);
    int nx = atoi(argv[6]);
    int ny = atoi(argv[7]);
    int nz = atoi(argv[8]);
    int T = atoi(argv[9]);
    int seed = atoi(argv[10]);
    int F = atoi(argv[11]);
    double iso = atof(argv[12]);

   

    // Check for total number of processes
    if (size != px * py * pz)
    {
        if (rank == 0)
            printf("Number of processes does not match the grid dimensions\n");
        MPI_Finalize();
        return 1;
    }

    // find coordinate of current process in the 3D process grid
    int mx = rank % px;
    int my = (rank / px) % py;
    int mz = rank / (px * py);

    // find stencil radius for given d point stencil computation
    int stencilRadius = (d - 1) / 6;     // d=7 -> 1, d=13 -> 2

    // Declaration of neighbor arrays
    int left[stencilRadius], right[stencilRadius];
    int up[stencilRadius], down[stencilRadius];
    int front[stencilRadius], back[stencilRadius];


    // finding neighbors of process
    for (int i = 0; i < stencilRadius; i++)
    {
        int dist = i + 1;

        left[i] = (mx - dist >= 0) ? rank - dist : -1;
        right[i] = (mx + dist < px) ? rank + dist : -1;

        down[i] = (my - dist >= 0) ? rank - dist * px : -1;
        up[i] = (my + dist < py) ? rank + dist * px : -1;

        back[i] = (mz - dist >= 0) ? rank - dist * px * py : -1;
        front[i] = (mz + dist < pz) ? rank + dist * px * py : -1;
    }

    // local array sizes including halo cells
    int LX = nx + 2 * stencilRadius;
    int LY = ny + 2 * stencilRadius;
    int LZ = nz + 2 * stencilRadius;

    // allocate memory for arrays used 
    double *data = malloc(F * LX * LY * LZ * sizeof(double));
    double *next = malloc(F * LX * LY * LZ * sizeof(double));

    // initialize arrays to zero
    for (int i = 0; i < F * LX * LY * LZ; i++)
    {
        data[i] = 0.0;
        next[i] = 0.0;
    }

    // Initialize values with random numbers with given seed
    srand(seed);
    int arrSize = nx * ny * nz;

    for (int i = 0; i < F; i++)
    {
        for (int j = 0; j < arrSize; j++)
        {
            int x = (j % nx) + stencilRadius;
            int y = ((j / nx) % ny) + stencilRadius;
            int z = (j / (nx * ny)) + stencilRadius;

            int idx = i * LX * LY * LZ + x * LY * LZ + y * LZ + z;
            data[idx] = (double)rand() * (rank + 1) / (110426.0 + i + j);
        }
    }


    // Initializing array for data counts for each field
    long long *fieldCounts = malloc(F * sizeof(long long));

    // Create MPI datatypes for halo exchange
    MPI_Datatype xPlaneType;
    MPI_Datatype YPlaneType;
    MPI_Datatype ZPlaneType;

    // Create derived datatype for yz-plane (for x-direction exchange)
    MPI_Type_vector(ny, nz, LZ, MPI_DOUBLE, &xPlaneType);
    MPI_Type_commit(&xPlaneType);

    // Create derived datatype for xz-plane (for y-direction exchange)
    MPI_Type_vector(nx, nz, LY * LZ, MPI_DOUBLE, &YPlaneType);
    MPI_Type_commit(&YPlaneType);

    // Create derived datatype for z-line (for z-direction exchange)
    MPI_Type_vector(ny, stencilRadius, LZ, MPI_DOUBLE, &ZPlaneType);
    MPI_Type_commit(&ZPlaneType);

    // Allocate memory for MPI requests
    MPI_Request *reqsX = malloc(4 * stencilRadius * sizeof(MPI_Request));
    MPI_Request *reqsY = malloc(4 * stencilRadius * sizeof(MPI_Request));
    MPI_Request *reqsZ = malloc(4 * nx * sizeof(MPI_Request));

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Main loop
    for (int t = 0; t <= T; t++)
    {
        for (int f = 0; f < F; f++)
        {
            int fieldOffset = f * LX * LY * LZ;

            
            // Step 1. Halo Exchange
           
            // Step 1.1 X direction
            int countX = 0;
            for (int i = 0; i < stencilRadius; i++)
            {
                // Calculate offsets for sending and receiving in left and right directions
                int leftRecvOffset  = fieldOffset + i * LY * LZ + stencilRadius * LZ + stencilRadius;
                int leftSendOffset  = fieldOffset + (stencilRadius + i) * LY * LZ + stencilRadius * LZ + stencilRadius;
                int rightSendOffset = fieldOffset + (nx + i) * LY * LZ + stencilRadius * LZ + stencilRadius;
                int rightRecvOffset = fieldOffset + (nx + stencilRadius + i) * LY * LZ + stencilRadius * LZ + stencilRadius;

                if (left[0] != -1) // if neighbor exists
                    MPI_Irecv(&data[leftRecvOffset], 1, xPlaneType, left[0], 10 + i, MPI_COMM_WORLD, &reqsX[countX++]);
                if (right[0] != -1)
                    MPI_Irecv(&data[rightRecvOffset], 1, xPlaneType, right[0], 20 + i, MPI_COMM_WORLD, &reqsX[countX++]);
                
                if (left[0] != -1)
                    MPI_Isend(&data[leftSendOffset], 1, xPlaneType, left[0], 20 + i, MPI_COMM_WORLD, &reqsX[countX++]);
                if (right[0] != -1)
                    MPI_Isend(&data[rightSendOffset], 1, xPlaneType, right[0], 10 + i, MPI_COMM_WORLD, &reqsX[countX++]);
            }
            MPI_Waitall(countX, reqsX, MPI_STATUSES_IGNORE);

            // Step 1.2 Y direction
            int countY = 0;
            for (int i = 0; i < stencilRadius; i++)
            {
                // Calculate offsets for sending and receiving in up and down directions
                int downRecvOffset = fieldOffset + stencilRadius * LY * LZ + i * LZ + stencilRadius;
                int downSendOffset = fieldOffset + stencilRadius * LY * LZ + (stencilRadius + i) * LZ + stencilRadius;
                int upSendOffset   = fieldOffset + stencilRadius * LY * LZ + (ny + i) * LZ + stencilRadius;
                int upRecvOffset   = fieldOffset + stencilRadius * LY * LZ + (ny + stencilRadius + i) * LZ + stencilRadius;

                if (down[0] != -1)
                    MPI_Irecv(&data[downRecvOffset], 1, YPlaneType, down[0], 30 + i, MPI_COMM_WORLD, &reqsY[countY++]);
                if (up[0] != -1)
                    MPI_Irecv(&data[upRecvOffset], 1, YPlaneType, up[0], 40 + i, MPI_COMM_WORLD, &reqsY[countY++]);
                
                if (down[0] != -1)
                    MPI_Isend(&data[downSendOffset], 1, YPlaneType, down[0], 40 + i, MPI_COMM_WORLD, &reqsY[countY++]);
                if (up[0] != -1)
                    MPI_Isend(&data[upSendOffset], 1, YPlaneType, up[0], 30 + i, MPI_COMM_WORLD, &reqsY[countY++]);
            }
            MPI_Waitall(countY, reqsY, MPI_STATUSES_IGNORE);

            // Step 1.3 Z direction
            int countZ = 0;
            for (int i = 0; i < nx; i++)
            {
                // Calculate offsets for sending and receiving in front and back directions
                int backRecvOffset  = fieldOffset + (stencilRadius + i) * LY * LZ + stencilRadius * LZ + 0;
                int backSendOffset  = fieldOffset + (stencilRadius + i) * LY * LZ + stencilRadius * LZ + stencilRadius;
                int frontSendOffset = fieldOffset + (stencilRadius + i) * LY * LZ + stencilRadius * LZ + nz;
                int frontRecvOffset = fieldOffset + (stencilRadius + i) * LY * LZ + stencilRadius * LZ + nz + stencilRadius;

                if (back[0] != -1)
                    MPI_Irecv(&data[backRecvOffset], 1, ZPlaneType, back[0], 50 + i, MPI_COMM_WORLD, &reqsZ[countZ++]);
                if (front[0] != -1)
                    MPI_Irecv(&data[frontRecvOffset], 1, ZPlaneType, front[0], 60 + i, MPI_COMM_WORLD, &reqsZ[countZ++]);
                
                if (back[0] != -1)
                    MPI_Isend(&data[backSendOffset], 1, ZPlaneType, back[0], 60 + i, MPI_COMM_WORLD, &reqsZ[countZ++]);
                if (front[0] != -1)
                    MPI_Isend(&data[frontSendOffset], 1, ZPlaneType, front[0], 50 + i, MPI_COMM_WORLD, &reqsZ[countZ++]);
            }
            MPI_Waitall(countZ, reqsZ, MPI_STATUSES_IGNORE);

            
            // Step 2. Isovalue Count
            // Not counting in first iteration as values are not updated yet
            if (t > 0)
            {
                long long localCount = 0;

                for (int x = stencilRadius; x < stencilRadius + nx; x++)
                {
                    for (int y = stencilRadius; y < stencilRadius + ny; y++)
                    {
                        for (int z = stencilRadius; z < stencilRadius + nz; z++)
                        {
                            int c = f * LX * LY * LZ + x * LY * LZ + y * LZ + z;
                            
                            if (x < stencilRadius + nx - 1 || right[0] != -1) // Check if we can compare with right neighbor
                                if ((data[c] - iso) * (data[c + LY * LZ] - iso) < 0) // Sign change indicates isovalue crossing
                                 localCount++;

                            if (y < stencilRadius + ny - 1 || up[0] != -1)
                                if ((data[c] - iso) * (data[c + LZ] - iso) < 0)
                                 localCount++;

                            if (z < stencilRadius + nz - 1 || front[0] != -1)
                                if ((data[c] - iso) * (data[c + 1] - iso) < 0)
                                 localCount++;
                        }
                    }
                }

                long long globalCount = 0;
                // MPI Reduce to get the total count across all processes (summing up local counts) at root (rank 0)
                MPI_Reduce(&localCount, &globalCount, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

                if (rank == 0)
                    fieldCounts[f] = globalCount;
            }

            
            // Step 3. Stencil Computation 
            
            if (t < T)  // Will not compute after the last iteration
            {
                for (int x = stencilRadius; x < stencilRadius + nx; x++)
                {
                    for (int y = stencilRadius; y < stencilRadius + ny; y++)
                    {
                        for (int z = stencilRadius; z < stencilRadius + nz; z++)
                        {
                            // index for the current grid point in the 1D array
                            int c = f * LX * LY * LZ + x * LY * LZ + y * LZ + z;
                            double sum = data[c];
                            int count = 1; // number of valid neighbors including itself

                            // Loop over stencil radius in all 6 directions
                            for (int r = 1; r <= stencilRadius; r++)
                            {
                                // Checking if neighbor exists
                                if (x - r >= stencilRadius || left[0] != -1)
                                {
                                    sum += data[c - r * LY * LZ];
                                    count++;
                                }
                                if (x + r < stencilRadius + nx || right[0] != -1)
                                {
                                    sum += data[c + r * LY * LZ];
                                    count++;
                                }
                                if (y - r >= stencilRadius || down[0] != -1)
                                {
                                    sum += data[c - r * LZ];
                                    count++;
                                }
                                if (y + r < stencilRadius + ny || up[0] != -1)
                                {
                                    sum += data[c + r * LZ];
                                    count++;
                                }
                                if (z - r >= stencilRadius || back[0] != -1)
                                {
                                    sum += data[c - r];
                                    count++;
                                }
                                if (z + r < stencilRadius + nz || front[0] != -1)
                                {
                                    sum += data[c + r];
                                    count++;
                                }
                            }

                            next[c] = sum / count;
                        }
                    }
                }
            }
        } 

        // Print counts for the completed timestep
        if (t > 0 && rank == 0)
        {
            for (int f = 0; f < F; f++)
            {
                if (f) printf(" ");
                printf("%lld", fieldCounts[f]);
            }
            printf("\n");
        }

        // Swap buffers for the next iteration
        if (t < T)
        {
            double *tmp = data;
            data = next;
            next = tmp;
        }
    }


    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    // Print total execution time
    if (rank == 0)
        printf("%lf\n", end - start);

    // Free all allocated resources

    MPI_Type_free(&xPlaneType);
    MPI_Type_free(&YPlaneType);
    MPI_Type_free(&ZPlaneType);

    free(reqsX);
    free(reqsY);
    free(reqsZ);

    free(data);
    free(next);
    free(fieldCounts);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}