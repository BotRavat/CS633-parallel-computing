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

    double *data_received = (double *)malloc(M * sizeof(double));
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
        data_received[i] = (double)rand() * (rank + 1) / 10000.0;
        bufferUpdatedForD1[i] = data_received[i];
        bufferUpdatedForD2[i] = data_received[i];
        dataReceivedByD1[i] = data_received[i];
        dataReceivedByD2[i] = data_received[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int iter = 0; iter < T; iter++)
    {
        int validSenderForD1 = (rank + D1 < size);
        int validSenderForD2 = (rank + D2 < size);

        int recvFromD1 = (rank - D1 >= 0);
        int recvFromD2 = (rank - D2 >= 0);

        int even = (rank % 2 == 0);

        /* ---------------- EVEN SEND FIRST ---------------- */
        if (even)
        {
            if (validSenderForD1)
            {
                MPI_Send(bufferUpdatedForD1, M, MPI_DOUBLE,
                         rank + D1, 0, MPI_COMM_WORLD);
            }

            if (validSenderForD2)
            {
                MPI_Send(bufferUpdatedForD2, M, MPI_DOUBLE,
                         rank + D2, 1, MPI_COMM_WORLD);
            }
        }

        /* ---------------- Receiver at D1 ---------------- */
        if (recvFromD1)
        {
            MPI_Recv(buffer, M, MPI_DOUBLE,
                     rank - D1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < M; i++)
            {
                buffer[i] = buffer[i] * buffer[i];
            }

            MPI_Send(buffer, M, MPI_DOUBLE,
                     rank - D1, 2, MPI_COMM_WORLD);
        }

        /* ---------------- Receiver at D2 ---------------- */
        if (recvFromD2)
        {
            MPI_Recv(buffer, M, MPI_DOUBLE,
                     rank - D2, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < M; i++)
            {
                buffer[i] = log(buffer[i]);
            }

            MPI_Send(buffer, M, MPI_DOUBLE,
                     rank - D2, 3, MPI_COMM_WORLD);
        }

        /* ---------------- ODD SEND AFTER RECEIVE ---------------- */
        if (!even)
        {
            if (validSenderForD1)
            {
                MPI_Send(bufferUpdatedForD1, M, MPI_DOUBLE,
                         rank + D1, 0, MPI_COMM_WORLD);
            }

            if (validSenderForD2)
            {
                MPI_Send(bufferUpdatedForD2, M, MPI_DOUBLE,
                         rank + D2, 1, MPI_COMM_WORLD);
            }
        }

        /* ---------------- Sender receives results ---------------- */
        if (validSenderForD1)
        {
            MPI_Recv(buffer, M, MPI_DOUBLE,
                     rank + D1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < M; i++)
            {
                data_received[i] = buffer[i];
                dataReceivedByD1[i] = buffer[i];
            }
        }

        if (validSenderForD2)
        {
            MPI_Recv(buffer, M, MPI_DOUBLE,
                     rank + D2, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < M; i++)
            {
                data_received[i] += buffer[i];
                dataReceivedByD2[i] = buffer[i];
            }
        }

        if (validSenderForD1)
        {
            for (int i = 0; i < M; i++)
            {
                bufferUpdatedForD1[i] = (unsigned long long)data_received[i] % 100000;
            }
        }
        if (validSenderForD2)
        {
            for (int i = 0; i < M; i++)
            {
                bufferUpdatedForD2[i] = data_received[i] * 100000;
            }
        }
    }

    /* ---------------- FINAL MAX + TIME (include final comm) ---------------- */

    double max_val_at_D1 = -1e18;
    double max_val_at_D2 = -1e18;

    if (rank + D1 < size)
        max_val_at_D1 = get_max_from_array(dataReceivedByD1, M);

    if (rank + D2 < size)
        max_val_at_D2 = get_max_from_array(dataReceivedByD2, M);

    if (rank != 0)
    {
        finalMaxValues[0] = max_val_at_D1;
        finalMaxValues[1] = max_val_at_D2;
        MPI_Send(finalMaxValues, 2, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);
    }
    else
    {
        double globalMaxD1 = max_val_at_D1;
        double globalMaxD2 = max_val_at_D2;

        for (int src = 1; src < size; src++)
        {
            MPI_Recv(finalMaxValues, 2, MPI_DOUBLE,
                     src, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (finalMaxValues[0] > globalMaxD1)
                globalMaxD1 = finalMaxValues[0];

            if (finalMaxValues[1] > globalMaxD2)
                globalMaxD2 = finalMaxValues[1];
        }

        double t_end = MPI_Wtime();
        printf("%lf %lf %lf\n", globalMaxD1, globalMaxD2, t_end - t_start);
    }

    free(data_received);
    free(buffer);
    free(bufferUpdatedForD1);
    free(bufferUpdatedForD2);
    free(dataReceivedByD1);
    free(dataReceivedByD2);
    free(finalMaxValues);

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
9999800001.000000 34.538756 0.539791
9999800001.000000 34.538756 0.538447
9999800001.000000 34.538756 0.538109
9999800001.000000 34.538756 0.538284
9999800001.000000 34.538756 0.538316
9999800001.000000 34.538756 2.115922
9999800001.000000 34.538756 2.115258
9999800001.000000 34.538756 2.114936
9999800001.000000 34.538756 2.116085
9999800001.000000 34.538756 2.115190


*/