#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

/*
Returns maximum value in an array of length n
*/
double maxFromArray(double *arr, int n)
{
    double maxValue = DBL_MIN;
    for (int i = 0; i < n; i++)
    {
        if (arr[i] > maxValue)
            maxValue = arr[i];
    }
    return maxValue;
}

/*
Our approach is to implement in two batches: batch 1 for communication with ranks at distance D1 and and batch 2 for ranks at distance D2
Each batch has two phases to cover alternating sender/receiver groups.
Group size is D1 for batch 1 and D2 for batch 2.
In each phase Sender and Receiver groups are selected by (rank / D) % 2, where D is the distance for the batch.
In each phase, sender ranks send data to receiver ranks, receiver ranks process data and send back to sender ranks, and sender ranks update their local state based on received data.
After the two batches, each valid sender rank sends its final max values for D1 and D2 to rank 0, which computes the final max values and prints the result.
*/
int main(int argc, char *argv[])
{
    int rank, size;
    int M, D1, D2, T, seed;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*
     * Expected command line arguments:
     * argv[1] = M    (buffer length)
     * argv[2] = D1    distance of rank
     * argv[3] = D2    distance of rank
     * argv[4] = T    (number of iterations)
     * argv[5] = seed (random seed)
     */
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

    /*
       Allocate buffers:
        buffer: temporary computation buffer
        bufferUpdatedForD1 / bufferUpdatedForD2: buffers that hold the updated values that will be sent to D1 and D2
        dataReceivedByD1 / dataReceivedByD2: buffer that receives data from sender for reciever ranks at D1 and D2 distance
        dataReceivedFromD1 / dataReceivedFromD2: buffer at sender ranks that receives data processed by ranks at D1 and D2 distance
        finalMaxValues: two-value array sent to rank 0 at the end (max for D1, max for D2)
       */
    double *buffer = (double *)malloc(M * sizeof(double));
    double *bufferUpdatedForD1 = (double *)malloc(M * sizeof(double));
    double *bufferUpdatedForD2 = (double *)malloc(M * sizeof(double));
    double *dataReceivedByD1 = (double *)malloc(M * sizeof(double));
    double *dataReceivedFromD1 = (double *)malloc(M * sizeof(double));

    double *dataReceivedByD2 = (double *)malloc(M * sizeof(double));
    double *dataReceivedFromD2 = (double *)malloc(M * sizeof(double));
    double *finalMaxValues = (double *)malloc(2 * sizeof(double));

    /*
    Initialization of values
    Each rank uses same seed: values remains same across runs
    but introduced multiplication by (rank+1) so that each rank will have different initial value
    */
    srand(seed);
    for (int i = 0; i < M; i++)
    {
        bufferUpdatedForD1[i] = (double)rand() * (rank + 1) / 10000.0;
        bufferUpdatedForD2[i] = bufferUpdatedForD1[i];
        dataReceivedByD1[i] = bufferUpdatedForD1[i];
        dataReceivedByD2[i] = bufferUpdatedForD2[i];
    }

    // Timing code starts (before T iterations loop)
    double startTime = MPI_Wtime();

    // Main iteration loop
    for (int iter = 0; iter < T; iter++)
    {
        // Two batches one for D1 and second for D2, each batch has 2 phases
        /*
         Batch1 Phase1: distance D1
         for D1=2 (in our submission reports):
         sender ranks = 0,1,4,5,8,9,...
         receiver ranks = 2,3,6,7,10,11,...
         */

        int isSenderD1P1 = ((rank / D1) % 2 == 0) && (rank + D1 < size);
        int isReceiverD1P1 = ((rank / D1) % 2 == 1) && (rank - D1 >= 0);

        // Batch1 Phase1 distance D1
        // Step1: sender ranks send bufferUpdatedForD1 to receiver ranks at D1 distance
        if (isSenderD1P1)
        {
            MPI_Send(bufferUpdatedForD1, M, MPI_DOUBLE,
                     rank + D1, rank + D1, MPI_COMM_WORLD);
        }
        /* Step2: receiver ranks receive data into dataReceivedByD1,
           then process data and update by squaring,
           and send back to sender ranks
        */
        if (isReceiverD1P1)
        {
            // Step 2.0: receiver receives data into dataReceivedByD1
            MPI_Recv(dataReceivedByD1, M, MPI_DOUBLE,
                     rank - D1, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Step 2.1: receiver updated data by squaring each element
            for (int i = 0; i < M; i++)
            {
                buffer[i] = dataReceivedByD1[i] * dataReceivedByD1[i];
            }
            // Step 2.2: receiver sends processed data back to sender
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

        /*Batch1 Phase2 distance D1
           Sender and receiver roles are swapped (including case for rank 0 (cannot receive in this phase) and last reciever rank (cannot send in this phase)),
           but communication pattern remains same
           for D1=2:
           sender ranks = 2,3,6,7,10,11,...
           receiver ranks = 4,5,8,9,...
        */

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

        /*
         Batch2 Phase1: distance D2
         for D2=4 (in our submission reports):
         sender ranks = 0,1,2,3,8,9,10,11,...
         receiver ranks = 4,5,6,7,12,13,14,15,...
         */
        int isSenderD2P1 = ((rank / D2) % 2 == 0) && (rank + D2 < size);
        int isReceiverD2P1 = ((rank / D2) % 2 == 1) && (rank - D2 >= 0);

        /*
        batch2 phase1
        Step1: sender ranks send bufferUpdatedForD2 to receiver ranks at D2 distance
        */

        if (isSenderD2P1)
        {
            MPI_Send(bufferUpdatedForD2, M, MPI_DOUBLE,
                     rank + D2, rank + D2, MPI_COMM_WORLD);
        }
        /* Step2: receiver ranks receive data into dataReceivedByD2,
           then process data and update by applying log(),
           and send back to sender ranks
        */
        if (isReceiverD2P1)
        {
            // Step 2.0: receiver receives data into dataReceivedByD2
            MPI_Recv(dataReceivedByD2, M, MPI_DOUBLE,
                     rank - D2, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Step 2.1: receiver applies log() on each element
            for (int i = 0; i < M; i++)
            {
                buffer[i] = log(dataReceivedByD2[i]);
            }
            for (int i = 0; i < M; i++)
            {
                buffer[i] = log(dataReceivedByD2[i]);
            }
            // Step 2.2: receiver sends processed data back to sender
            MPI_Send(buffer, M, MPI_DOUBLE, rank - D2, rank - D2, MPI_COMM_WORLD);
        }
        // Step 3: sender receives processed data into dataReceivedFromD2 and updates bufferUpdatedForD2 by multiplying by 100000
        if (isSenderD2P1)
        {
            // Step 3.0: sender receives processed data into dataReceivedFromD2
            MPI_Recv(dataReceivedFromD2, M, MPI_DOUBLE,
                     rank + D2, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < M; i++)
            {
                bufferUpdatedForD2[i] = dataReceivedFromD2[i] * 100000;
            }
        }

        /*
         Batch2 Phase2 distance D2
         Sender and receiver roles are swapped (including case for rank 0 (cannot receive in this phase) and last reciever rank (cannot send in this phase)),
         but communication pattern remains same
         for D2=4:
         sender ranks = 4,5,6,7,12,13,14,15,...
         receiver ranks = 8,9,10,11,...
        */
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

    // final max value calculation and communication to rank 0
    // each valid sender will send final received max D1 and max D2  value to rank 0

    if (rank != 0)
    {
        if (rank + D1 < size)
        {
            // valid sender for D1 and D2 both: finds max values for both D1 and D2
            if (rank + D2 < size)
            {
                finalMaxValues[0] = fmax(DBL_MIN, maxFromArray(bufferUpdatedForD1, M));
                finalMaxValues[1] = fmax(DBL_MIN, maxFromArray(bufferUpdatedForD2, M));
            }
            else
            // valid sender for D1 but not for D2: finds max value for D1 and sets D2 max value to DBL_MIN
            {
                finalMaxValues[0] = fmax(DBL_MIN, maxFromArray(bufferUpdatedForD1, M));
                finalMaxValues[1] = DBL_MIN;
            }

            // send max values to rank 0
            MPI_Send(finalMaxValues, 2, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
        }
    }
    else
    {
        double max_val_at_D1 = DBL_MIN;
        double max_val_at_D2 = DBL_MIN;
        // Rank 0 includes its own values if valid.
        if (0 + D1 < size)
        {
            max_val_at_D1 = fmax(max_val_at_D1, maxFromArray(bufferUpdatedForD1, M));
            if (0 + D2 < size)
            {
                max_val_at_D2 = fmax(max_val_at_D2, maxFromArray(bufferUpdatedForD2, M));
            }
        }
        /* Many to one communication:
        Rank 0 receives max values from all valid sender ranks
        and updates final max values for D1 and D2 among all received values and its own value
        */
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

        // Timing code ends (after T iterations and extraction of final max values)
        double endTime = MPI_Wtime();
        double executionTime = endTime - startTime;
        // Print final result: Single line containing three numbers: <maximumD1> <maximumD2> <time>
        printf("%lf %lf %lf\n", max_val_at_D1, max_val_at_D2, executionTime);
    }

    // Free all allocated memory
    free(buffer);
    free(dataReceivedFromD1);
    free(dataReceivedFromD2);
    free(bufferUpdatedForD1);
    free(bufferUpdatedForD2);
    free(dataReceivedByD1);
    free(dataReceivedByD2);
    free(finalMaxValues);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
