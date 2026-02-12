#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double get_max_from_array(double *arr, int n)
{
    double max_val = -1e18;
    for (int i = 0; i < n; i++)
    {
        if (arr[i] > max_val)
            max_val = arr[i];
    }
    return max_val;
}

int main(int argc, char *argv[])
{
    int rank, size;
    int M, D1, D2, T, seed;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 6)
    {
        if (rank == 0)
            printf("Usage: %s M D1 D2 T seed\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    M = atoi(argv[1]);
    D1 = atoi(argv[2]);
    D2 = atoi(argv[3]);
    T = atoi(argv[4]);
    seed = atoi(argv[5]);


    double *buffer = (double *)malloc(M * sizeof(double));
    double *bufferUpdatedForD1 = (double *)malloc(M * sizeof(double));
    double *bufferUpdatedForD2 = (double *)malloc(M * sizeof(double));
    double *dataReceivedByD1 = (double *)malloc(M * sizeof(double));
    double *dataReceivedByD2 = (double *)malloc(M * sizeof(double));
    double *finalMaxValues = (double *)malloc(2 * sizeof(double));

    // Initialize data
    srand(seed);
    for (int i = 0; i < M; i++)
    {
        bufferUpdatedForD1[i] = (double)rand() * (rank + 1) / 10000.0;
        bufferUpdatedForD2[i] =  bufferUpdatedForD1[i];
        dataReceivedByD1[i] = bufferUpdatedForD1[i];
        dataReceivedByD2[i] = bufferUpdatedForD2[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int iter = 0; iter < T; iter++)
    {

        int validSenderForD1 = (rank + D1 < size);
        int validSenderForD2 = (rank + D2 < size);

        /* ---------------- Sender side ---------------- */
        if (validSenderForD1)
        {
            MPI_Send(bufferUpdatedForD1, M, MPI_DOUBLE,
                     rank + D1, rank + D1, MPI_COMM_WORLD);
        }

        if (validSenderForD2)
        {
            MPI_Send(bufferUpdatedForD2, M, MPI_DOUBLE,
                     rank + D2, rank + D2, MPI_COMM_WORLD);
        }

        /* ---------------- Receiver at D1 ---------------- */
        if (rank - D1 >= 0)
        {
            MPI_Recv(buffer, M, MPI_DOUBLE,
                     rank - D1, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < M; i++)
            {
                buffer[i] = buffer[i] * buffer[i];
            }

            MPI_Send(buffer, M, MPI_DOUBLE,
                     rank - D1, rank - D1, MPI_COMM_WORLD);
        }

        /* ---------------- Receiver at D2 ---------------- */
        if (rank - D2 >= 0)
        {
            MPI_Recv(buffer, M, MPI_DOUBLE,
                     rank - D2, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < M; i++)
            {
                buffer[i] = log(buffer[i]);
            }

            MPI_Send(buffer, M, MPI_DOUBLE,
                     rank - D2, rank - D2, MPI_COMM_WORLD);
        }

        /* ---------------- Sender receives results ---------------- */
        if (validSenderForD1)
        {
            MPI_Recv(buffer, M, MPI_DOUBLE,
                     rank + D1, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < M; i++)
            {
                dataReceivedByD1[i] = buffer[i];
            }
        }

        if (validSenderForD2)
        {
            MPI_Recv(buffer, M, MPI_DOUBLE,
                     rank + D2, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < M; i++)
            {
                dataReceivedByD2[i] = buffer[i];
            }
        }

        if (validSenderForD1)
        {
            for (int i = 0; i < M; i++)
            {
                bufferUpdatedForD1[i] = (unsigned long long)dataReceivedByD1[i] % 100000;
            }
        }
        if (validSenderForD2)
        {
            for (int i = 0; i < M; i++)
            {
                bufferUpdatedForD2[i] = dataReceivedByD2[i] * 100000;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;

    if (rank != 0)
    {
        if (rank + D1 < size)
        {
            if (rank + D2 < size)
            {
                finalMaxValues[0] = fmax(-1e9, get_max_from_array(dataReceivedByD1, M));
                finalMaxValues[1] = fmax(-1e9, get_max_from_array(dataReceivedByD2, M));
            }
            else
            {
                finalMaxValues[0] = fmax(-1e9, get_max_from_array(dataReceivedByD1, M));
                finalMaxValues[1] = -1e9;
            }

            MPI_Send(finalMaxValues, 2, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
        }
    }
    else
    {
        // double max_val_at_D1 = fmax(-1e9, dataReceivedByD1[0]);
        // double max_val_at_D2 = fmax(-1e9, dataReceivedByD2[0]);
        double max_val_at_D1 = -1e9;
        double max_val_at_D2 = -1e9;

        for (int src = 1; src < size; src++)
        {
            if (src + D2 < size || src + D1 < size)
            {
                MPI_Recv(finalMaxValues, 2, MPI_DOUBLE,
                         src, src, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                max_val_at_D1 = fmax(max_val_at_D1, finalMaxValues[0]);
                max_val_at_D2 = fmax(max_val_at_D2, finalMaxValues[1]);
            }
        }
        printf("Input parameters: M=%d D1=%d D2=%d T=%d seed=%d\n", M, D1, D2, T, seed);
        printf("%lf %lf %lf\n", max_val_at_D1, max_val_at_D2, local_time);
    }

    free(buffer);

    MPI_Finalize();
    return 0;
}




/*

#parameters
D1=2
D2=4
T=5
SEED=1000
MSIZES=(262144 1048576)
P_COUNTS=(8)

==========================================
9999800001.000000 34.538756 0.277905
9999800001.000000 34.538756 0.277033
9999800001.000000 34.538756 0.277117
9999800001.000000 34.538756 0.277058
9999800001.000000 34.538756 0.276825
9999800001.000000 34.538756 1.095148
9999800001.000000 34.538756 1.094458
9999800001.000000 34.538756 1.094350
9999800001.000000 34.538756 1.094181
9999800001.000000 34.538756 1.094769





*/