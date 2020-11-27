	/*
* file:   utilities.c
* Implementation of project's functions
*
* authors: Charalampos Papadakis (9128), Portokalidis Stavros (9334)
* emails: papadakic@ece.auth.gr , stavport@ece.auth.gr
* date:   2019-12-01
*/


#include <stdio.h>
#include <stdlib.h>
#include "knnring.h"
#include <math.h>


void swapElement(double **first, double  **second){
	double  *temp = *first;
	*first = *second;
	*second = temp;
}

double SumRow(double *array, int numOfColumns, int row) {
  double result=0;

  for(int j=0; j<numOfColumns; j++){
    result += pow(*(array+row*numOfColumns+j),2);
  }

  return result;
}


void qselect(double *tArray,int *index, int len, int k) {
	#	define SWAP(a, b) { tmp = tArray[a]; tArray[a] = tArray[b]; tArray[b] = tmp; }
  #	define SWAPINDEX(a, b) { tmp = index[a]; index[a] = index[b]; index[b] = tmp; }
	int i, st;
	double tmp;

	for (st = i = 0; i < len - 1; i++) {
		if (tArray[i] > tArray[len-1]) continue;
		SWAP(i, st);
    SWAPINDEX(i,st);
		st++;
	}
	SWAP(len-1, st);
  SWAPINDEX(len-1,st);
  if(k < st){
    qselect(tArray, index,st, k);
  }
  else if(k > st){
    qselect(tArray + st, index + st, len - st, k - st);
  }
  if (k == st){
    return ;
  }
  return ;
}



void quicksort(double *array, int *idx, int first, int last){
   int i, j, pivot;
   double  temp;

   if(first<last){
      pivot=first;
      i=first;
      j=last;

      while(i<j){
         while(array[i]<=array[pivot]&&i<last)
            i++;
         while(array[j]>array[pivot])
            j--;
         if(i<j){
            temp=array[i];
            array[i]=array[j];
            array[j]=temp;

            temp=idx[i];
            idx[i]=idx[j];
            idx[j]=temp;
         }
      }

      temp=array[pivot];
      array[pivot]=array[j];
      array[j]=temp;

      temp=idx[pivot];
      idx[pivot]=idx[j];
      idx[j]=temp;

      quicksort(array,idx,first,j-1);
      quicksort(array,idx,j+1,last);

   }
}


knnresult changeResult(knnresult result,knnresult tempResult,int offset,int newOff){
  double *y = (double *)malloc(result.m*result.k*sizeof(double));
  int *yidx = (int *)malloc(result.m*result.k*sizeof(int));

	if(y==NULL || yidx==NULL){
		exit(1);
	}

  int myCounter , newCounter , allCounter;
  for(int i=0; i<result.m; i++){
    myCounter=0, newCounter=0, allCounter=0;
    while (allCounter<result.k) {
        if (*(result.ndist + i*result.k + myCounter) < *(tempResult.ndist + i*result.k+ newCounter)){
          *(y+i*result.k+allCounter) = *(result.ndist+ i*result.k+myCounter);
          *(yidx+i*result.k+allCounter) = *(result.nidx+i*result.k+myCounter) + offset*result.m;
          allCounter++;
          myCounter++;
        }
        else{
          *(y+i*result.k+allCounter) = *(tempResult.ndist+i*result.k+newCounter);
          *(yidx+i*result.k+allCounter) = *(tempResult.nidx+i*result.k+newCounter) + newOff*result.m  ;
          allCounter++;
          newCounter++;
        }
    }
  }
  for(int i=0; i<result.m; i++){
    for(int j = 0 ; j <result.k ; j++){
      *(result.ndist+i*result.k+j) = *(y+i*result.k+j);
      *(result.nidx+i*result.k+j)= *(yidx+i*result.k+j);
    }
  }

  return result;
}
