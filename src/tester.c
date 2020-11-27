/*!
  \file   tester.c
  \brief  Validate kNN ring implementation.

  \author Dimitris Floros
  \date   2019-11-13
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include "knnring.h"





#ifndef TESTER_HELPER_H
#define TESTER_HELPER_H

/* #define VERBOSE */

static char * STR_CORRECT_WRONG[] = {"WRONG", "CORRECT"};

// =================
// === UTILITIES ===
// =================

// double distColMajor(double *X, double *Y,
//                     int i, int j,
//                     int d, int n, int m){
//
//   /* compute distance */
//   double dist = 0;
//   for (int l = 0; l < d; l++){
//     dist += ( X[l*n+i] - Y[l*m+j] ) * ( X[l*n+i] - Y[l*m+j] );
//   }
//
//   return sqrt(dist);
// }

double distRowMajor(double *X, double *Y,
                    int i, int j,
                    int d, int n, int m){

  /* compute distance */
  double dist = 0;
  for (int l = 0; l < d; l++){
    dist += ( X[l+i*d] - Y[l+j*d] ) * ( X[l+i*d] - Y[l+j*d] );
  }

  return sqrt(dist);
}



// ==================
// === VALIDATION ===
// ==================

// //! kNN validator (col major)
// /*!
//    The function asserts correctness of the kNN results by:
//      (i)   Checking that reported distances are correct
//      (ii)  Validating that distances are sorted in non-decreasing order
//      (iii) Ensuring there are no other points closer than the kth neighbor
// */
// int validateResultColMajor( knnresult knnres,
//                             double * corpus, double * query,
//                             int n, int m, int d, int k ) {
//
//   /* loop through all query points */
//   for (int j = 0; j < m; j++ ){
//
//     /* max distance so far (equal to kth neighbor after nested loop) */
//     double maxDist = -1;
//
//     /* mark all distances as not computed */
//     int * visited = (int *) calloc( n, sizeof(int) );
//
//     /* go over reported k nearest neighbors */
//     for (int i = 0; i < k; i++ ){
//
//       /* keep list of visited neighbors */
//       visited[ knnres.nidx[i*m + j] ] = 1;
//
//       /* get distance to stored index */
//       double distxy = distColMajor( corpus, query, knnres.nidx[i*m + j], j, d, n, m );
//
//       /* make sure reported distance is correct */
//       if ( fabs( knnres.ndist[i*m + j] - distxy ) > 1e-8 ) return 0;
//
//       /* distances should be non-decreasing */
//       if ( knnres.ndist[i*m + j] < maxDist ) return 0;
//
//       /* update max neighbor distance */
//       maxDist = knnres.ndist[i*m + j];
//
//     } /* for (k) -- reported nearest neighbors */
//
//     /* now maxDist should have distance to kth neighbor */
//
//     /* check all un-visited points */
//     for (int i = 0; i < n; i++ ){
//
//       /* check only (n-k) non-visited nodes */
//       if (!visited[i]){
//
//         /* get distance to unvisited vertex */
//         double distxy = distColMajor( corpus, query, i, j, d, n, m );
//
//         /* point cannot be closer than kth distance */
//         if ( distxy < maxDist ) return 0;
//
//       } /* if (!visited[i]) */
//
//     } /* for (i) -- unvisited notes */
//
//     /* deallocate memory */
//     free( visited );// double distColMajor(double *X, double *Y,
// //                     int i, int j,
// //                     int d, int n, int m){
// //
// //   /* compute distance */
// //   double dist = 0;
// //   for (int l = 0; l < d; l++){
// //     dist += ( X[l*n+i] - Y[l*m+j] ) * ( X[l*n+i] - Y[l*m+j] );
// //   }
// //
// //   return sqrt(dist);
// // }
//
//   } /* for (j) -- query points */
//
//   /* return */
//   return 1;
//
// }


//! kNN validator (row major)
/*!
   The function asserts correctness of the kNN results by:
     (i)   Checking that reported distances are correct
     (ii)  Validating that distances are sorted in non-decreasing order
     (iii) Ensuring there are no other points closer than the kth neighbor
*/
int validateResultRowMajor( knnresult knnres,
                            double * corpus, double * query,
                            int n, int m, int d, int k ) {

  /* loop through all query points */
  for (int j = 0; j < m; j++ ){

    /* max distance so far (equal to kth neighbor after nested loop) */
    double maxDist = -1;

    /* mark all distances as not computed */
    int * visited = (int *) calloc( n, sizeof(int) );

    /* go over reported k nearest neighbors */
    for (int i = 0; i < k; i++ ){

      /* keep list of visited neighbors */
      visited[ knnres.nidx[i + j*k] ] = 1;

      /* get distance to stored index */
      double distxy = distRowMajor( corpus, query, knnres.nidx[i + j*k], j, d, n, m );

      /* make sure reported distance is correct */
      if ( fabs( knnres.ndist[i + j*k] - distxy ) > 1e-8 ) return 0;

      /* distances should be non-decreasing */
      if ( knnres.ndist[i + j*k] < maxDist ) return 0;

      /* update max neighbor distance */
      maxDist = knnres.ndist[i + j*k];

    } /* for (k) -- reported nearest neighbors */

    /* now maxDist should have distance to kth neighbor */

    /* check all un-visited points */
    for (int i = 0; i < n; i++ ){

      /* check only (n-k) non-visited nodes */
      if (!visited[i]){

        /* get distance to unvisited vertex */
        double distxy = distRowMajor( corpus, query, i, j, d, n, m );

        /* point cannot be closer than kth distance */
        if ( distxy < maxDist ) return 0;

      } /* if (!visited[i]) */

    } /* for (i) -- unvisited notes */

    /* deallocate memory */
    free( visited );

  } /* for (j) -- query points */

  /* return */
  return 1;

}





#endif





int main()
{

  int n=891;                    // corpus
  int m=762;                    // query
  int d=7;                      // dimensions
  int k=13;                     // # neighbors
  int i,j;

  double  * corpus = (double * ) malloc( n*d * sizeof(double) );
  double  * query  = (double * ) malloc( m*d * sizeof(double) );

  for (int i=0;i<n*d;i++)
    corpus[i]= ((double)(rand()%100))/50;

  for (int i=0;i<m*d;i++)
    query[i]= ((double)(rand()%100))/50;

  knnresult knnres = kNN( corpus, query, n,m,d,k);

//   printf ("\n Matrix Corpus: \n");
// for (i=0; i<n; i++) {
//   for (j=0; j<d; j++) {
//     printf ("%10.2lf", *(corpus+j+i*d));
//   }
//   printf ("\n");
// }
//
// printf ("\n Matrix Query: \n");
// for (i=0; i<m; i++) {
//   for (j=0; j<d; j++) {
//     printf ("%10.2lf", *(query+i*d+j));
//   }
//   printf ("\n");
// }
// //
// // printf("\n\n");
// printf ("\n Matrix DISTANCE: \n");
// for (i=0; i<m; i++) {
//   for (j=0; j<k; j++) {
//     printf ("%lf    ", *(knnres.ndist+i*k+j));
//   }
//   printf ("\n");
// }
//
// printf ("\n Matrix INDECES: \n");
// for (i=0; i<m; i++) {
//   for (j=0; j<k; j++) {
//     printf ("%10.2d", *(knnres.nidx+i*k+j));
//   }
//   printf ("\n");
// }

  int isValid = validateResultRowMajor( knnres, corpus, query, n, m, d, k );

  printf("Tester validation: %s NEIGHBORS\n", STR_CORRECT_WRONG[isValid]);

  free( corpus );
  free( query );

  return 0;

}
