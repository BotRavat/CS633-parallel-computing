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

void process_phase(int rank, int D, int M, int isSender, int isReceiver, double *bufferUpdated, double *dataReceived, double *buffer, int isD1)
{
    if (isSender)
    {
        MPI_Send(bufferUpdated, M, MPI_DOUBLE,
                 rank + D, rank + D, MPI_COMM_WORLD);
    }
    if (isReceiver)
    {
        MPI_Recv(dataReceived, M, MPI_DOUBLE,
                 rank - D, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < M; i++)
        {
            if (isD1)
            {
                buffer[i] = dataReceived[i] * dataReceived[i];
            }
            else
            {
                buffer[i] = log(dataReceived[i]);
                // buffer[i] = 2*dataReceived[i];
            }
        }
        MPI_Send(buffer, M, MPI_DOUBLE,
                 rank - D, rank - D, MPI_COMM_WORLD);
    }
    if (isSender)
    {
        MPI_Recv(dataReceived, M, MPI_DOUBLE,
                 rank + D, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < M; i++)
        {
            if (isD1)
            {
                bufferUpdated[i] = (unsigned long long)dataReceived[i] % 100000;
                // bufferUpdated[i] = 2*dataReceived[i];
            }
            else
            {
                bufferUpdated[i] = dataReceived[i] * 100000;
                // bufferUpdated[i] = dataReceived[i] * dataReceived[i];
            }
        }
    }
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
    double finalMaxValues[2];

    srand(seed);
    for (int i = 0; i < M; i++)
    {
        bufferUpdatedForD1[i] = (double)rand() * (rank + 1) / 100000.0;
        // bufferUpdatedForD1[i] = (double)(rank + 1) * (i + 1);
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

        // batch1 phase1
        process_phase(rank, D1, M, isSenderD1P1, isReceiverD1P1, bufferUpdatedForD1, dataReceivedByD1, buffer, 1);

        // batch1 phase2
        // sender ranks = 2,3,6,7,10,11,...
        // receiver ranks = 4,5,8,9,12,13,...
        int isSenderD1P2 = (((rank / D1) % 2) == 1) && (rank + D1 < size);
        int isReceiverD1P2 = (((rank / D1) % 2) == 0) && (rank - D1 >= 0);

        process_phase(rank, D1, M, isSenderD1P2, isReceiverD1P2, bufferUpdatedForD1, dataReceivedByD1, buffer, 1);

        // batch2   distance =d2=4
        // sender ranks = 0,1,2,3,8,9,10,11,...
        // receiver ranks = 4,5,6,7,12,13,14,15,...
        int isSenderD2P1 = ((rank / D2) % 2 == 0) && (rank + D2 < size);
        int isReceiverD2P1 = ((rank / D2) % 2 == 1) && (rank - D2 >= 0);

        // batch2 phase1
        process_phase(rank, D2, M, isSenderD2P1, isReceiverD2P1, bufferUpdatedForD2, dataReceivedByD2, buffer, 0);

        // batch2 phase2
        int isSenderD2P2 = (((rank / D2) % 2) == 1) && (rank + D2 < size);
        int isReceiverD2P2 = (((rank / D2) % 2) == 0) && (rank - D2 >= 0);

        process_phase(rank, D2, M, isSenderD2P2, isReceiverD2P2, bufferUpdatedForD2, dataReceivedByD2, buffer, 0);
    }

    // final max and time
    // each valid sender will send final received max d1 and max d2  value to rank 0

    if (rank != 0)
    {
        if (rank + D1 < size)
        {
            if (rank + D2 < size)
            {
                finalMaxValues[0] = fmax(-1e18, get_max_from_array(dataReceivedByD1, M));
                finalMaxValues[1] = fmax(-1e18, get_max_from_array(dataReceivedByD2, M));
            }
            else
            {
                finalMaxValues[0] = fmax(-1e18, get_max_from_array(dataReceivedByD1, M));
                finalMaxValues[1] = -1e18;
            }

            MPI_Send(finalMaxValues, 2, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
        }
    }
    else
    {
        // double max_val_at_D1 = fmax(-1e18, dataReceivedByD1[0]);
        // double max_val_at_D2 = fmax(-1e18, dataReceivedByD2[0]);
        double max_val_at_D1 = -1e18;
        double max_val_at_D2 = -1e18;

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
        printf("%lf %lf %lf\n", max_val_at_D1, max_val_at_D2, local_time);
    }

    free(buffer);
    free(bufferUpdatedForD1);
    free(bufferUpdatedForD2);
    free(dataReceivedByD1);
    free(dataReceivedByD2);

    MPI_Finalize();
    return 0;
}