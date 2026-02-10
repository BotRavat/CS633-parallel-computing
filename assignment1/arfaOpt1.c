#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int M = atoi(argv[1]);
    int D1 = atoi(argv[2]);
    int D2 = atoi(argv[3]);
    int T  = atoi(argv[4]);
    int seed = atoi(argv[5]);

    double *data = malloc(M*sizeof(double));
    double *bufD1 = malloc(M*sizeof(double));
    double *bufD2 = malloc(M*sizeof(double));

    double *bufferUpdatedForD1 = malloc(M*sizeof(double));
    double *bufferUpdatedForD2 = malloc(M*sizeof(double));
    double *data_received = malloc(M*sizeof(double));

    srand(seed);
    for(int i=0;i<M;i++) {
        data[i] = (double)rand()*(rank+1)/10000.0;
        bufferUpdatedForD1[i] = data[i];
        bufferUpdatedForD2[i] = data[i];
        data_received[i] = data[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for(int iter=0; iter<T; iter++) {

        int sendD1 = (rank + D1 < P);
        int sendD2 = (rank + D2 < P);

        int recvD1 = (rank - D1 >= 0);
        int recvD2 = (rank - D2 >= 0);

        // -------------------------------
        // STEP 1: RECEIVE FIRST (parallel safe)
        // -------------------------------

        if(recvD1)
            MPI_Recv(bufD1, M, MPI_DOUBLE,
                     rank-D1, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

        if(recvD2)
            MPI_Recv(bufD2, M, MPI_DOUBLE,
                     rank-D2, 1, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

        // -------------------------------
        // STEP 2: COMPUTE AT RECEIVERS
        // -------------------------------

        if(recvD1)
            for(int i=0;i<M;i++)
                bufD1[i] = bufD1[i]*bufD1[i];

        if(recvD2)
            for(int i=0;i<M;i++)
                bufD2[i] = log(bufD2[i]);

        // -------------------------------
        // STEP 3: SEND BACK RESULTS
        // -------------------------------

        if(recvD1)
            MPI_Send(bufD1, M, MPI_DOUBLE,
                     rank-D1, 2, MPI_COMM_WORLD);

        if(recvD2)
            MPI_Send(bufD2, M, MPI_DOUBLE,
                     rank-D2, 3, MPI_COMM_WORLD);

        // -------------------------------
        // STEP 4: SEND TO RECEIVERS
        // -------------------------------

        if(sendD1)
            MPI_Send(bufferUpdatedForD1, M, MPI_DOUBLE,
                     rank+D1, 0, MPI_COMM_WORLD);

        if(sendD2)
            MPI_Send(bufferUpdatedForD2, M, MPI_DOUBLE,
                     rank+D2, 1, MPI_COMM_WORLD);

        // -------------------------------
        // STEP 5: RECEIVE RESULTS
        // -------------------------------

        if(sendD1)
            MPI_Recv(bufD1, M, MPI_DOUBLE,
                     rank+D1, 2, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

        if(sendD2)
            MPI_Recv(bufD2, M, MPI_DOUBLE,
                     rank+D2, 3, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

        // -------------------------------
        // STEP 6: UPDATE DATA (MATCHES seqOpt LOGIC)
        // -------------------------------

        for(int i=0;i<M;i++) {
            data_received[i] = 0.0;
            if(sendD1) data_received[i] = bufD1[i];
            if(sendD2) data_received[i] += bufD2[i];
        }

        if(sendD1) {
            for(int i=0;i<M;i++)
                bufferUpdatedForD1[i] = (unsigned long long)data_received[i] % 100000;
        }

        if(sendD2) {
            for(int i=0;i<M;i++)
                bufferUpdatedForD2[i] = data_received[i] * 100000;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    /* -------------------------------
       FINAL MAX + PRINTING
       ------------------------------- */

    double localMaxD1 = -1e18;
    double localMaxD2 = -1e18;

    int sendD1 = (rank + D1 < P);
    int sendD2 = (rank + D2 < P);

    if(sendD1) {
        for(int i=0;i<M;i++)
            if(bufD1[i] > localMaxD1) localMaxD1 = bufD1[i];
    }

    if(sendD2) {
        for(int i=0;i<M;i++)
            if(bufD2[i] > localMaxD2) localMaxD2 = bufD2[i];
    }

    if(rank != 0) {
        MPI_Send(&localMaxD1, 1, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD);
        MPI_Send(&localMaxD2, 1, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD);
    }

    if(rank == 0) {

        double globalMaxD1 = localMaxD1;
        double globalMaxD2 = localMaxD2;

        for(int r=1;r<P;r++) {
            double t1, t2;
            MPI_Recv(&t1, 1, MPI_DOUBLE, r, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&t2, 1, MPI_DOUBLE, r, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if(t1 > globalMaxD1) globalMaxD1 = t1;
            if(t2 > globalMaxD2) globalMaxD2 = t2;
        }

        printf("%lf %lf %lf\n", globalMaxD1, globalMaxD2, end-start);
    }

    free(data);
    free(bufD1);
    free(bufD2);
    free(bufferUpdatedForD1);
    free(bufferUpdatedForD2);
    free(data_received);

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
9999800001.000000 34.538756 0.517978
9999800001.000000 34.538756 0.517109
9999800001.000000 34.538756 0.517356
9999800001.000000 34.538756 0.517413
9999800001.000000 34.538756 0.517256
9999800001.000000 34.538756 2.025272
9999800001.000000 34.538756 2.024817
9999800001.000000 34.538756 2.024735
9999800001.000000 34.538756 2.024596
9999800001.000000 34.538756 2.025191


*/