#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double get_max_from_array(double *arr, int n) {
    double max_val = -1e18;
    for (int i = 0; i < n; i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    return max_val;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int M, D1, D2, T, seed;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 6) {
        if (rank == 0) printf("Usage: %s M D1 D2 T seed\n", argv[0]);
        MPI_Finalize();
        return 0;
    }
    
    M = atoi(argv[1]);
    D1 = atoi(argv[2]);
    D2 = atoi(argv[3]);
    T = atoi(argv[4]);
    seed = atoi(argv[5]);
    
    // Separate buffers for D1 and D2 to avoid conflicts
    double *data_received = (double *)malloc(M * sizeof(double));
    double *bufferD1 = (double *)malloc(M * sizeof(double));
    double *bufferD2 = (double *)malloc(M * sizeof(double));
    double *bufferUpdatedForD1 = (double *)malloc(M * sizeof(double));
    double *bufferUpdatedForD2 = (double *)malloc(M * sizeof(double));
    double *dataReceivedByD1 = (double *)malloc(M * sizeof(double));
    double *dataReceivedByD2 = (double *)malloc(M * sizeof(double));
    double *finalMaxValues = (double *)malloc(2 * sizeof(double));
    
    // Initialize data
    srand(seed);
    for (int i = 0; i < M; i++) {
        data_received[i] = (double)rand() * (rank + 1) / 10000.0;
        bufferUpdatedForD1[i] = data_received[i];
        bufferUpdatedForD2[i] = data_received[i];
        dataReceivedByD1[i] = data_received[i];
        dataReceivedByD2[i] = data_received[i];
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();
    
    int validSenderForD1 = (rank + D1 < size);
    int validSenderForD2 = (rank + D2 < size);
    int validReceiverFromD1 = (rank - D1 >= 0);
    int validReceiverFromD2 = (rank - D2 >= 0);
    
    for (int iter = 0; iter < T; iter++) {
        
        /* ================================================================
           CLUSTERING STRATEGY: Separate sends and receives to avoid deadlock
           and allow parallel communication on independent paths.
           
           Pattern: All processes first receive, then send back results.
           This ensures senders and receivers are synchronized properly.
           ================================================================ */
        
        /* ---------------- PHASE 1: Receivers get data and process ---------------- */
        
        // D1 path: Receive from rank-D1, process, and send back
        if (validReceiverFromD1) {
            MPI_Recv(bufferD1, M, MPI_DOUBLE, rank - D1, rank, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Process D1: square operation
            for (int i = 0; i < M; i++) {
                bufferD1[i] = bufferD1[i] * bufferD1[i];
            }
            
            MPI_Send(bufferD1, M, MPI_DOUBLE, rank - D1, rank - D1, MPI_COMM_WORLD);
        }
        
        // D2 path: Receive from rank-D2, process, and send back
        if (validReceiverFromD2) {
            MPI_Recv(bufferD2, M, MPI_DOUBLE, rank - D2, rank, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Process D2: log operation
            for (int i = 0; i < M; i++) {
                bufferD2[i] = log(bufferD2[i]);
            }
            
            MPI_Send(bufferD2, M, MPI_DOUBLE, rank - D2, rank - D2, MPI_COMM_WORLD);
        }
        
        /* ---------------- PHASE 2: Senders send data and receive results ---------------- */
        
        // D1 path: Send to rank+D1 and receive result
        if (validSenderForD1) {
            MPI_Send(bufferUpdatedForD1, M, MPI_DOUBLE, rank + D1, rank + D1, 
                    MPI_COMM_WORLD);
            
            MPI_Recv(bufferD1, M, MPI_DOUBLE, rank + D1, rank, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int i = 0; i < M; i++) {
                data_received[i] = bufferD1[i];
                dataReceivedByD1[i] = bufferD1[i];
            }
        }
        
        // D2 path: Send to rank+D2 and receive result
        if (validSenderForD2) {
            MPI_Send(bufferUpdatedForD2, M, MPI_DOUBLE, rank + D2, rank + D2, 
                    MPI_COMM_WORLD);
            
            MPI_Recv(bufferD2, M, MPI_DOUBLE, rank + D2, rank, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int i = 0; i < M; i++) {
                // data_received[i] += bufferD2[i];
                dataReceivedByD2[i] = bufferD2[i];
            }
        }
        
        /* ---------------- Update buffers for next iteration ---------------- */
        
        if (validSenderForD1) {
            for (int i = 0; i < M; i++) {
                bufferUpdatedForD1[i] = (unsigned long long)dataReceivedByD1[i] % 100000;
            }
        }
        
        if (validSenderForD2) {
            for (int i = 0; i < M; i++) {
                bufferUpdatedForD2[i] = dataReceivedByD2[i] * 100000;
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;
    
    /* ---------------- Gather results at rank 0 ---------------- */
    if (rank != 0) {
        if (rank + D1 < size) {
            if (rank + D2 < size) {
                finalMaxValues[0] = fmax(-1e9, get_max_from_array(dataReceivedByD1, M));
                finalMaxValues[1] = fmax(-1e9, get_max_from_array(dataReceivedByD2, M));
            } else {
                finalMaxValues[0] = fmax(-1e9, get_max_from_array(dataReceivedByD1, M));
                finalMaxValues[1] = -1e9;
            }
            MPI_Send(finalMaxValues, 2, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
        }
    } else {
        double max_val_at_D1 = -1e9;
        double max_val_at_D2 = -1e9;
        
        for (int src = 1; src < size; src++) {
            if (src + D2 < size || src + D1 < size) {
                MPI_Recv(finalMaxValues, 2, MPI_DOUBLE, src, src, 
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                max_val_at_D1 = fmax(max_val_at_D1, finalMaxValues[0]);
                max_val_at_D2 = fmax(max_val_at_D2, finalMaxValues[1]);
            }
        }
        
        printf("%lf %lf %lf\n", max_val_at_D1, max_val_at_D2, local_time);
    }
    
    free(data_received);
    free(bufferD1);
    free(bufferD2);
    free(bufferUpdatedForD1);
    free(bufferUpdatedForD2);
    free(dataReceivedByD1);
    free(dataReceivedByD2);
    free(finalMaxValues);
    
    MPI_Finalize();
    return 0;
}






/*

Running for P=8 M=262144 D1=2 D2=4 T=10 SEED=1000
9987204096.000000 14.163601 0.157591
9987204096.000000 14.163601 0.156224
9987204096.000000 14.163601 0.156408
9987204096.000000 14.163601 0.156267
9987204096.000000 14.163601 0.156015

Running for P=8 M=1048576 D1=2 D2=4 T=10 SEED=1000
9987204096.000000 14.163601 0.155717
9987204096.000000 14.163601 0.156220
9987204096.000000 14.163601 0.156091
9987204096.000000 14.163601 0.155942
9987204096.000000 14.163601 0.156248



Running for P=16 M=262144 D1=2 D2=4 T=10 SEED=1000
9987204096.000000 14.163601 0.248498
9987204096.000000 14.163601 0.246854
9987204096.000000 14.163601 0.246575
9987204096.000000 14.163601 0.248248
9987204096.000000 14.163601 0.247376

Running for P=16 M=1048576 D1=2 D2=4 T=10 SEED=1000
9987204096.000000 14.163601 0.247164
9987204096.000000 14.163601 0.247660
9987204096.000000 14.163601 0.246265
9987204096.000000 14.163601 0.251034
9987204096.000000 14.163601 0.247587



Running for P=32 M=262144 D1=2 D2=4 T=10 SEED=1000
9987204096.000000 14.163601 0.329456
9987204096.000000 14.163601 0.327546
9987204096.000000 14.163601 0.326220
9987204096.000000 14.163601 0.328262
9987204096.000000 14.163601 0.322413

Running for P=32 M=1048576 D1=2 D2=4 T=10 SEED=1000
9987204096.000000 14.163601 0.327080
9987204096.000000 14.163601 0.326716
9987204096.000000 14.163601 0.327596
9987204096.000000 14.163601 0.325186
9987204096.000000 14.163601 0.323494





*/