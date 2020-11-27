/*
* file:   knnring_mpi_asyc.c
* Iplemantation of knnring asychronous verision
*
* authors: Charalampos Papadakis (9128) , Portokalidis Stavros (9334)
* emails: papadakic@ece.auth.gr , stavport@ece.auth.gr
* date:   2019-12-01
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cblas.h"
#include "knnring.h"
#include <mpi.h>
#include "utilities.h"

knnresult kNN(double * X , double * Y , int n , int m , int d , int k) {

  knnresult result;
  result.k = k;
  result.m = m;
  result.nidx = NULL;
  result.ndist = NULL;
  int pid, numtasks;
  MPI_Comm_rank(MPI_COMM_WORLD,&pid);
  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);


  double * distance;
  int *indeces;
  double alpha=-2.0, beta=0.0;
  int lda=d, ldb=d, ldc=m, i, j;
  int counter = 0;

  distance = (double *) malloc( n * m *sizeof(double));
  indeces= (int*)malloc(m * n  *sizeof(int));

  if(distance==NULL || indeces==NULL){
    exit(1);
  }

  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++) {
      *(indeces+i*n+j)=j;
    }
  }

  cblas_dgemm(CblasRowMajor , CblasNoTrans , CblasTrans , n, m , d , alpha , X , lda , Y , ldb , beta, distance , ldc);

  double * xRow = (double *) calloc(n,sizeof(double));
  double * yRow = (double *) calloc(m,sizeof(double));
  double * transD = (double *) malloc(m*n*sizeof(double));

  for(int i=0; i<n; i++){
    for(int j=0; j<d; j++){
      xRow[i] += (*(X+i*d+j)) * (*(X+i*d+j));
    }
  }
  for(int i=0; i<m; i++){
    for(int j=0; j<d; j++){
      yRow[i] += (*(Y+i*d+j)) * (*(Y+i*d+j));
    }
  }
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      *(distance + i*m + j) += xRow[i] + yRow[j];
      if(*(distance + i*m + j) < 0.00000001){
        *(distance + i*m + j) = 0;
      }
      else{
        *(distance + i*m + j) = sqrt( *(distance + i*m + j) );
      }
    }
  }

  // calculate transpose matrix
  if(transD==NULL){
    exit(1);
}

  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      *(transD + j*n + i ) = *(distance + i*m + j );
    }
  }

  double * final = (double *) malloc(m*k * sizeof(double));
  int * finalIdx = (int *) malloc (m * k * sizeof(int));
  double * temp = (double *) malloc(n * sizeof(double));
  int * tempIdx = (int *) malloc (n * sizeof(int));

  if(final==NULL || finalIdx==NULL || temp==NULL || tempIdx==NULL){
    exit(1);
  }

  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      *(temp+j) = *(transD+i*n+j);
      *(tempIdx+j)= *(indeces+i*n+j);
    }
    qselect(temp,tempIdx,n,k);
    quicksort(temp, tempIdx,0,k);
    for(int j=0; j<k; j++){
      *(final+i*k+j) = temp[j];
      *(finalIdx+i*k+j) = tempIdx[j];
    }
  }

  free(temp);
  free(tempIdx);
  free(distance);
  free(indeces);

  result.ndist = final;
  result.nidx = finalIdx;

  return result;
}

knnresult distrAllkNN(double * X , int n , int d , int k ) {

  int numtasks , pid ;
  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&pid);
  MPI_Request request[3];
  MPI_Status status;


  int *idx =(int *)malloc(n*k*sizeof(int));
  double * dist = (double *) malloc(n * k * sizeof(double));
  double *buffer = (double *) malloc(n * d * sizeof(double));
  double *myElements = (double *) malloc(n * d * sizeof(double));
  double *otherElements = (double *) malloc(n * d * sizeof(double));

   if(idx==NULL || dist==NULL || buffer==NULL || myElements==NULL || otherElements==NULL){
     exit(1);
   }


  knnresult result ;
  knnresult tempResult  ;

  result.m=n;
  result.k=k;
  idx = result.nidx;
  dist = result.ndist;


  myElements = X;

  int counter= 2;
  int newNormaliseVar , normaliseVar;

  clock_t time1;

  MPI_Barrier(MPI_COMM_WORLD);

  time1=clock();

  if(pid%2){
      MPI_Isend(myElements , n*d , MPI_DOUBLE, (pid + 1)%numtasks , 0 , MPI_COMM_WORLD , &request[0] ); //non blocking send
      MPI_Irecv(otherElements , n*d , MPI_DOUBLE, pid - 1 , 0 , MPI_COMM_WORLD, &request[1]); //non blocking receive
      result = kNN(myElements,myElements,n,n,d,k);
      normaliseVar = (numtasks+pid-1)%numtasks;
      newNormaliseVar = (numtasks + normaliseVar-1)%numtasks;
      MPI_Wait(&request[1],&status);
      while(counter<numtasks){
        MPI_Isend(otherElements , n*d , MPI_DOUBLE, (pid + 1)%numtasks , 0 , MPI_COMM_WORLD , &request[2] );
        MPI_Irecv(buffer , n*d , MPI_DOUBLE, pid - 1 , 0 , MPI_COMM_WORLD, &request[1]);
          tempResult = kNN(otherElements ,  myElements, n , n , d ,k );

          if(counter == 2 ){
          result = changeResult( result, tempResult, normaliseVar, newNormaliseVar);
          }
          else{
            newNormaliseVar = (numtasks + newNormaliseVar-1)%numtasks;
            result = changeResult( result, tempResult, 0, newNormaliseVar);
          }
        MPI_Wait(&request[1],&status);
        MPI_Wait(&request[2],&status);
        swapElement(&otherElements,&buffer);
          counter++;
      }

      tempResult = kNN(otherElements ,  myElements, n , n , d ,k );
      if(numtasks!=2){
        newNormaliseVar = (numtasks + newNormaliseVar-1)%numtasks;
        normaliseVar=0;
      }
      result = changeResult( result, tempResult, normaliseVar, newNormaliseVar);
}
  else{
      MPI_Isend(myElements , n*d , MPI_DOUBLE, (pid + 1)%numtasks , 0 , MPI_COMM_WORLD , &request[0] );
      MPI_Irecv(otherElements , n*d , MPI_DOUBLE, pid - 1 , 0 , MPI_COMM_WORLD, &request[1]);

      result = kNN(myElements,myElements,n,n,d,k);
      normaliseVar = (numtasks+pid-1)%numtasks;
      newNormaliseVar = (numtasks + normaliseVar-1)%numtasks;
      MPI_Wait(&request[1],&status);

      while(counter<numtasks){
        MPI_Isend(otherElements , n*d , MPI_DOUBLE, (pid + 1)%numtasks , 0 , MPI_COMM_WORLD , &request[2] );
        MPI_Irecv(buffer , n*d , MPI_DOUBLE, pid - 1 , 0 , MPI_COMM_WORLD, &request[1]);

        tempResult = kNN(otherElements ,  myElements, n , n , d ,k );

        if(counter == 2 ){
          result = changeResult( result, tempResult, normaliseVar, newNormaliseVar);
        }
        else{
          newNormaliseVar = (numtasks + newNormaliseVar-1)%numtasks;
          result = changeResult( result, tempResult, 0, newNormaliseVar);
        }

        MPI_Wait(&request[1],&status);
        MPI_Wait(&request[2],&status);
        swapElement(&otherElements,&buffer);
        counter++;
      }

      tempResult = kNN(otherElements ,  myElements, n , n , d ,k );
      if(numtasks!=2){
        newNormaliseVar = (numtasks + newNormaliseVar-1)%numtasks;
        normaliseVar=0;
      }
      result = changeResult( result, tempResult, normaliseVar, newNormaliseVar);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  time1 = clock() - time1;
double totalTime = ((double) time1) / CLOCKS_PER_SEC;
double averageTotalTime = 0;
MPI_Reduce( & totalTime, & averageTotalTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

if (pid == 0) {
  printf("Total time taken for V2 :  %lf\n", totalTime / (numtasks));

}

  double min=result.ndist[1];
  double max=result.ndist[0];
  for(int i=0; i <n*k; i++){
    if(result.ndist[i]>max){
      max = result.ndist[i];
    }
    if(result.ndist[i]<min && result.ndist[i]!=0){
      min = result.ndist[i];
    }
  }


  double globalMin;
  double globalMax;

  MPI_Reduce(&min, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0 , MPI_COMM_WORLD);
  MPI_Reduce(&max, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0 , MPI_COMM_WORLD);

  if(pid==0)
    printf("Global MAX : %lf, Global MIN : %lf  \n " , globalMax , globalMin );



  return result;
}
