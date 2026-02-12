#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc,char *argv[]) {

MPI_Init(&argc,&argv);

int rank,P;
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD,&P);

int M=atoi(argv[1]);
int D1=atoi(argv[2]);
int D2=atoi(argv[3]);
int T =atoi(argv[4]);
int seed=atoi(argv[5]);

double *current=malloc(M*sizeof(double));
double *next=malloc(M*sizeof(double));
double *bufD1=malloc(M*sizeof(double));
double *bufD2=malloc(M*sizeof(double));

srand(seed+rank);
for(int i=0;i<M;i++)
current[i]=(double)rand()*(rank+1)/10000.0;

int sendD1=(rank+D1<P);
int sendD2=(rank+D2<P);
int recvD1=(rank-D1>=0);
int recvD2=(rank-D2>=0);

MPI_Barrier(MPI_COMM_WORLD);
double start=MPI_Wtime();

for(int iter=0;iter<T;iter++){

int even=(rank%2==0);

/* EVEN SEND FIRST */
if(even){
if(sendD1) MPI_Send(current,M,MPI_DOUBLE,rank+D1,0,MPI_COMM_WORLD);
if(sendD2) MPI_Send(current,M,MPI_DOUBLE,rank+D2,1,MPI_COMM_WORLD);
}

/* RECEIVE + COMPUTE */
if(recvD1){
MPI_Recv(bufD1,M,MPI_DOUBLE,rank-D1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
for(int i=0;i<M;i++) bufD1[i]*=bufD1[i];
MPI_Send(bufD1,M,MPI_DOUBLE,rank-D1,2,MPI_COMM_WORLD);
}

if(recvD2){
MPI_Recv(bufD2,M,MPI_DOUBLE,rank-D2,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
for(int i=0;i<M;i++) bufD2[i]=log(bufD2[i]+1e-9);
MPI_Send(bufD2,M,MPI_DOUBLE,rank-D2,3,MPI_COMM_WORLD);
}

/* ODD SEND AFTER RECEIVE */
if(!even){
if(sendD1) MPI_Send(current,M,MPI_DOUBLE,rank+D1,0,MPI_COMM_WORLD);
if(sendD2) MPI_Send(current,M,MPI_DOUBLE,rank+D2,1,MPI_COMM_WORLD);
}

/* RECEIVE RESULTS */
if(sendD1)
MPI_Recv(bufD1,M,MPI_DOUBLE,rank+D1,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

if(sendD2)
MPI_Recv(bufD2,M,MPI_DOUBLE,rank+D2,3,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

/* UPDATE */
for(int i=0;i<M;i++){
double sum=0;
if(sendD1) sum+=bufD1[i];
if(sendD2) sum+=bufD2[i];
next[i]=sum;
}

/* SWAP BUFFERS */
double *tmp=current;
current=next;
next=tmp;
}

MPI_Barrier(MPI_COMM_WORLD);
double end=MPI_Wtime();

double localMaxD1=-1e18;
double localMaxD2=-1e18;

/* compute local max only if valid sender */
if(sendD1){
for(int i=0;i<M;i++)
if(bufD1[i]>localMaxD1)
localMaxD1=bufD1[i];
}

if(sendD2){
for(int i=0;i<M;i++)
if(bufD2[i]>localMaxD2)
localMaxD2=bufD2[i];
}

/* send to rank 0 */
if(rank!=0){
MPI_Send(&localMaxD1,1,MPI_DOUBLE,0,10,MPI_COMM_WORLD);
MPI_Send(&localMaxD2,1,MPI_DOUBLE,0,11,MPI_COMM_WORLD);
}

double globalMaxD1=localMaxD1;
double globalMaxD2=localMaxD2;

if(rank==0){
for(int r=1;r<P;r++){
double t1,t2;
MPI_Recv(&t1,1,MPI_DOUBLE,r,10,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
MPI_Recv(&t2,1,MPI_DOUBLE,r,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

if(t1>globalMaxD1) globalMaxD1=t1;
if(t2>globalMaxD2) globalMaxD2=t2;
}

printf("%lf %lf %lf\n",
globalMaxD1,
globalMaxD2,
end-start);
}

free(current);
free(next);
free(bufD1);
free(bufD2);

MPI_Finalize();
return 0;
}