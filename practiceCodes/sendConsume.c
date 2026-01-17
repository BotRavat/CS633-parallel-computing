#include<stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char *argv[])
{
  
  
    int myrank,size;
    MPI_Status status;
    double sTime, eTime, time;
  
      MPI_Init(&argc, &argv);
      int count=atoi(argv[1]);
      int buf[count];

    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
     for(int i=0;i<count;i++)
     {
        buf[i]=myrank +i;
     }
     if(myrank==0)
    {
     MPI_Send (buf,count,MPI_INT,1,1,MPI_COMM_WORLD);
    printf("%d \n 0 gg ",myrank);
     MPI_Send (buf,count,MPI_INT,1,2,MPI_COMM_WORLD);
    printf("%d \n",myrank);
    }
     else
     if(myrank==1)
{     MPI_Recv(buf,count,MPI_INT,0,1,MPI_COMM_WORLD,&status);  // sent message by same tag will be consumed by 1 recv 
     MPI_Recv(buf,count,MPI_INT,0,1,MPI_COMM_WORLD,&status);}  // due to tag mismatch it will recv the msg send with tag 1 as it was already recieved by previous recv

    printf("%d %d \n",myrank,count);

    MPI_Finalize();
}