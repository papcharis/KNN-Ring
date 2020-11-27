#ifndef KNNRING_H
#define KNNRING_H

typedef struct knnresult{
  int * nidx; 
  double * ndist;
  int m;
  int k;
} knnresult;

/*
 *Method kNN():
 *Description:
 *	This method takes a corpus set X and a query set Y and
 *	calculates the k nearest neighbors for each point of Y.
 *Return value:
 *	returns a kNNresult-type result with the k-nearest neighbors.
 */
knnresult kNN(double * X , double * Y , int n , int m, int d , int k);


/*
 *Method distrAllkNN():
 *Description:
 *	This method makes the k-nn ring problem with open MPI
 *      environment commands with multiple processes.
 *Return value: 
 *	Returns a knnresult-type result after comparing with every set 
 *	of points that have been received.
 */
knnresult distrAllkNN(double * X, int n, int d, int k);


#endif
