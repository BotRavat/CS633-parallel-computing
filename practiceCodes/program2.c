#include<stdio.h>
#include "mpi.h"

int main(int argc, char *argv[])
{
    MPI_Init(NULL,NULL);
  
    int size;

    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int rank;
   
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("HEllo I am rank %d out of %d processes\n",rank,size);

    MPI_Finalize();
}