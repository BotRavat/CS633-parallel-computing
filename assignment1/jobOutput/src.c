#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

double get_max_from_array(double *arr, int n)
{
    double max_val = DBL_MIN;
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
    double *dataReceivedFromD1 = (double *)malloc(M * sizeof(double));

    double *dataReceivedByD2 = (double *)malloc(M * sizeof(double));
    double *dataReceivedFromD2 = (double *)malloc(M * sizeof(double));
    double *finalMaxValues = (double *)malloc(2 * sizeof(double));

    srand(seed);
    for (int i = 0; i < M; i++)
    {
        bufferUpdatedForD1[i] = (double)rand() * (rank + 1) / 10000.0;
        bufferUpdatedForD2[i] = bufferUpdatedForD1[i];
        dataReceivedByD1[i] = bufferUpdatedForD1[i];
        dataReceivedByD2[i] = bufferUpdatedForD2[i];
    }

    // MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int iter = 0; iter < T; iter++)
    {

        // batch1   distance =d1=2
        // sender ranks = 0,1,4,5,8,9,...
        // receiver ranks = 2,3,6,7,10,11,...
        int isSenderD1P1 = ((rank / D1) % 2 == 0) && (rank + D1 < size);
        int isReceiverD1P1 = ((rank / D1) % 2 == 1) && (rank - D1 >= 0);

        // batch2   distance =d2=4
        // sender ranks = 0,1,2,3,8,9,10,11,...
        // receiver ranks = 4,5,6,7,12,13,14,15,...
        int isSenderD2P1 = ((rank / D2) % 2 == 0) && (rank + D2 < size);
        int isReceiverD2P1 = ((rank / D2) % 2 == 1) && (rank - D2 >= 0);

        // batch1 phase1

        if (isSenderD1P1)
        {
            MPI_Send(bufferUpdatedForD1, M, MPI_DOUBLE,
                     rank + D1, rank + D1, MPI_COMM_WORLD);
        }
        if (isReceiverD1P1)
        {
            MPI_Recv(dataReceivedByD1, M, MPI_DOUBLE,
                     rank - D1, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < M; i++)
            {
                buffer[i] = dataReceivedByD1[i] * dataReceivedByD1[i];
            }

            MPI_Send(buffer, M, MPI_DOUBLE,
                     rank - D1, rank - D1, MPI_COMM_WORLD);
        }
        if (isSenderD1P1)
        {
            MPI_Recv(dataReceivedFromD1, M, MPI_DOUBLE,
                     rank + D1, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < M; i++)
            {
                bufferUpdatedForD1[i] = (unsigned long long)dataReceivedFromD1[i] % 100000;
            }
        }

        // batch1 phase2
        int isSenderD1P2 = ((rank / D1) % 2 == 1) && (rank + D1 < size);
        int isReceiverD1P2 = ((rank / D1) % 2 == 0) && (rank - D1 >= 0);

        if (isSenderD1P2)
        {
            MPI_Send(bufferUpdatedForD1, M, MPI_DOUBLE,
                     rank + D1, rank + D1, MPI_COMM_WORLD);
        }
        if (isReceiverD1P2)
        {
            MPI_Recv(dataReceivedByD1, M, MPI_DOUBLE,
                     rank - D1, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < M; i++)
            {
                buffer[i] = dataReceivedByD1[i] * dataReceivedByD1[i];
            }
            MPI_Send(buffer, M, MPI_DOUBLE,
                     rank - D1, rank - D1, MPI_COMM_WORLD);
        }
        if (isSenderD1P2)
        {
            MPI_Recv(dataReceivedFromD1, M, MPI_DOUBLE,
                     rank + D1, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < M; i++)
            {
                bufferUpdatedForD1[i] = (unsigned long long)dataReceivedFromD1[i] % 100000;
            }
        }

        // batch2 phase1
        if (isSenderD2P1)
        {
            MPI_Send(bufferUpdatedForD2, M, MPI_DOUBLE,
                     rank + D2, rank + D2, MPI_COMM_WORLD);
        }
        if (isReceiverD2P1)
        {
            MPI_Recv(dataReceivedByD2, M, MPI_DOUBLE,
                     rank - D2, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < M; i++)
            {
                buffer[i] = log(dataReceivedByD2[i]);
            }
            MPI_Send(buffer, M, MPI_DOUBLE, rank - D2, rank - D2, MPI_COMM_WORLD);
        }
        if (isSenderD2P1)
        {
            MPI_Recv(dataReceivedFromD2, M, MPI_DOUBLE,
                     rank + D2, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < M; i++)
            {
                bufferUpdatedForD2[i] = dataReceivedFromD2[i] * 100000;
            }
        }

        // batch2 phase2
        int isSenderD2P2 = ((rank / D2) % 2 == 1) && (rank + D2 < size);
        int isReceiverD2P2 = ((rank / D2) % 2 == 0) && (rank - D2 >= 0);

        if (isSenderD2P2)
        {
            MPI_Send(bufferUpdatedForD2, M, MPI_DOUBLE,
                     rank + D2, rank + D2, MPI_COMM_WORLD);
        }
        if (isReceiverD2P2)
        {
            MPI_Recv(dataReceivedByD2, M, MPI_DOUBLE,
                     rank - D2, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < M; i++)
            {
                buffer[i] = log(dataReceivedByD2[i]);
            }
            MPI_Send(buffer, M, MPI_DOUBLE,
                     rank - D2, rank - D2, MPI_COMM_WORLD);
        }
        if (isSenderD2P2)
        {
            MPI_Recv(dataReceivedFromD2, M, MPI_DOUBLE,
                     rank + D2, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < M; i++)
            {
                bufferUpdatedForD2[i] = dataReceivedFromD2[i] * 100000;
            }
        }
    }

    // final max and time
    // each valid sender will send final received max d1 and max d2  value to rank 0

    if (rank != 0)
    {
        if (rank + D1 < size)
        {
            if (rank + D2 < size)
            {
                finalMaxValues[0] = fmax(DBL_MIN, get_max_from_array(bufferUpdatedForD1, M));
                finalMaxValues[1] = fmax(DBL_MIN, get_max_from_array(bufferUpdatedForD2, M));
            }
            else
            {
                finalMaxValues[0] = fmax(DBL_MIN, get_max_from_array(bufferUpdatedForD1, M));
                finalMaxValues[1] = DBL_MIN;
            }

            MPI_Send(finalMaxValues, 2, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
        }
    }
    else
    {
        // double max_val_at_D1 = fmax(DBL_MIN, dataReceivedByD1[0]);
        // double max_val_at_D2 = fmax(DBL_MIN, dataReceivedByD2[0]);
        double max_val_at_D1 = DBL_MIN;
        double max_val_at_D2 = DBL_MIN;

        if (0 + D1 < size)
        {
            max_val_at_D1 = fmax(max_val_at_D1, get_max_from_array(bufferUpdatedForD1, M));
            if (0 + D2 < size)
            {
                max_val_at_D2 = fmax(max_val_at_D2, get_max_from_array(bufferUpdatedForD2, M));
            }
        }

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

        double t_end = MPI_Wtime();
        double local_time = t_end - t_start;

        // printf("Input parameters: M=%d D1=%d D2=%d T=%d seed=%d\n", M, D1, D2, T, seed);
        printf("%lf %lf %lf\n", max_val_at_D1, max_val_at_D2, local_time);
    }

    free(buffer);
    free(dataReceivedFromD1);
    free(dataReceivedFromD2);
    free(bufferUpdatedForD1);
    free(bufferUpdatedForD2);
    free(dataReceivedByD1);
    free(dataReceivedByD2);
    free(finalMaxValues);

    MPI_Finalize();
    return 0;
}

