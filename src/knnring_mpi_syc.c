/*
* file:   knnring_mpi_syc.c
* Iplemantation of knnring sychronous verision
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
  double * transD = (double *)calloc(m*n,sizeof(double));

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
  free(xRow);
  free(yRow);
  // calculate transpose matrix
  if(transD==NULL){
    exit(1);
  }

  double temp2=0;
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      temp2 = *(distance + i*m + j );
      *(transD + j*n + i ) = temp2 ;
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

  int *idx =(int *)malloc(n*k*sizeof(int));
  double * dist = (double *) malloc(n * k * sizeof(double));

  knnresult result ;
  knnresult newResult  ;

  result.m=n;
  result.k=k;
  idx = result.nidx;
  dist = result.ndist;

  double *buffer = (double *) malloc(n * d * sizeof(double));
  double *myElements = (double *) malloc(n * d * sizeof(double));
  double *otherElements = (double *) malloc(n * d * sizeof(double));
  double *y = (double *)malloc(n*k*sizeof(double));
  int *yidx = (int *)malloc(n*k*sizeof(int));
  myElements = X;
  int counter= 2;
  int normaliseVar , newNormaliseVar ;

  clock_t time1, time2 ,total=0; //used for calculating calculations time,communication time and total time

  MPI_Barrier(MPI_COMM_WORLD); //waits for every process to come here

  time1=clock();

  //checks if pid is odd or even number and defines if it will send or receive first
  if (pid%2){
      MPI_Send(myElements , n*d , MPI_DOUBLE, (pid + 1)%numtasks , 0 , MPI_COMM_WORLD ); //sends to the next process-Blocks the program
      time2 = clock();
      result = kNN(myElements,myElements,n,n,d,k);
      total += clock() - time2;
      MPI_Recv(otherElements , n*d , MPI_DOUBLE, pid - 1 , 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE); //receives from previous process-Blocks the program
      time2 = clock();
      newResult = kNN(otherElements , myElements,  n , n , d ,k);
      normaliseVar = (numtasks+pid-1)%numtasks;
      newNormaliseVar = (numtasks+normaliseVar-1)%numtasks;
      result = changeResult( result, newResult, normaliseVar, newNormaliseVar);
      total += clock() - time2;

      while(counter<numtasks){
        MPI_Send(otherElements , n*d , MPI_DOUBLE, (pid + 1)%numtasks , 0 , MPI_COMM_WORLD );
        MPI_Recv(otherElements , n*d , MPI_DOUBLE, pid - 1 , 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        time2 = clock();
        newResult = kNN(otherElements ,  myElements, n , n , d ,k );
        newNormaliseVar = (numtasks + newNormaliseVar-1)%numtasks;
        result = changeResult( result, newResult, 0, newNormaliseVar);
        counter++;
        total += clock() - time2;
      }
  }
  else{
    MPI_Recv(otherElements , n*d , MPI_DOUBLE, (numtasks+pid- 1)%numtasks , 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    time2=clock();
    result = kNN(myElements,myElements,n,n,d,k);
    total += clock() - time2;
    MPI_Send(myElements , n*d , MPI_DOUBLE, (pid + 1)%numtasks , 0 , MPI_COMM_WORLD );
    time2 = clock();
    newResult = kNN(otherElements , myElements , n , n , d ,k);
    normaliseVar = (numtasks+pid-1)%numtasks;
    newNormaliseVar = (numtasks+normaliseVar-1)%numtasks;
    result = changeResult( result, newResult, normaliseVar, newNormaliseVar);
    total += clock() - time2;

    while(counter<numtasks){
      MPI_Recv(buffer , n*d , MPI_DOUBLE, (numtasks+pid- 1)%numtasks , 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(otherElements , n*d , MPI_DOUBLE, (pid + 1)%numtasks , 0 , MPI_COMM_WORLD );
      time2 = clock();
      swapElement(&otherElements, &buffer);
      newNormaliseVar = (numtasks + newNormaliseVar-1)%numtasks;
      newResult = kNN( otherElements , myElements,  n ,n , d ,k );
      result = changeResult( result, newResult, 0, newNormaliseVar);
      counter++;
      total += clock() - time2;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  time1 = clock() - time1;

  double calculationTime = ((double)total)/CLOCKS_PER_SEC;
  double totalTime = ((double)time1) / CLOCKS_PER_SEC;

  double averageCalcTime = 0;
  double averageTotalTime = 0;

  MPI_Reduce(&calculationTime,&averageCalcTime,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Reduce(&totalTime,&averageTotalTime,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

  if (pid == 0) {
   printf("Total time taken is: %lf\n", averageTotalTime/(numtasks));
   printf("Time taken for computing:  %lf\n", averageCalcTime / (numtasks));
   printf("Time taken for communication :  %lf\n", averageTotalTime/ (numtasks) - averageCalcTime / (numtasks));
 }

  //calculates the global Minimum and Maximum
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
    printf("Global MAX : %lf, Global MIN : %lf  \n "  , globalMax , globalMin );

  return result;
}
