/*
* file:   knnring_sequential.c
* Iplemantation of knnring sequential version
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
#include "utilities.h"



knnresult kNN(double * X , double * Y , int n , int m , int d , int k) {

  knnresult result; //the final result that will be returned
  result.k = k;
  result.m = m;
  result.nidx = NULL;
  result.ndist = NULL;

  //X: n * d
  //Y: m * d
  double * distance; //array of all computed distances
  int *indeces;  //array of indeces used for specific distance

  //parametres used for CBlas_dgemm function
  double alpha=-2.0, beta=0.0;
  int lda=d, ldb=d, ldc=m, i, j;

  int counter = 0;

  distance = (double *) calloc(n*m,sizeof(double));

  indeces= (int*)malloc(n*m*sizeof(int));

  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++) {
      *(indeces+i*n+j)=j;
    }
  }

  //multiplication of arrays X and trans Y
  cblas_dgemm(CblasRowMajor , CblasNoTrans , CblasTrans , n, m , d , alpha , X , lda , Y , ldb , beta, distance , ldc);


  for(int i=0; i<n; i++){
    double SumX =  SumRow(X, d, i);
    for(int j=0; j<m; j++){
      double SumY = SumRow(Y,d,j);
      *(distance + i*m + j ) += SumX + SumY;

      if(*(distance + i*m + j ) < 0.00000001){
        *(distance + i*m + j ) = 0;
      }
      else{
        *(distance + i*m + j ) = sqrt( *(distance + i*m + j ) );
      }
    }
  }

  // calculate transpose matrix
  double * transD = (double *) malloc(m*n*sizeof(double));
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      *(transD + j*n + i ) = *(distance + i*m + j );
    }
  }

  //distance = transD then delete transD
  for(int i=0; i<n*m; i++) {
    *(distance+i) = *(transD+i);
  }
  free(transD);

  double * final = (double *) malloc(m*k * sizeof(double));
  int * finalIdx = (int *) malloc (m * k * sizeof(int));
  double * temp = (double *) malloc(n * sizeof(double));
  int * tempIdx = (int *) malloc (n * sizeof(int));
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      *(temp+j) = *(distance+i*n+j);
      *(tempIdx+j)= *(indeces+i*n+j);
    }
    qselect(temp,tempIdx,n,k); //qselect in every line of the distances array
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
