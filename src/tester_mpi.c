/*!
  \file   tester.c
  \brief  Validate kNN ring implementation (MPI).

  \author Dimitris Floros
  \date   2019-11-25
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <mpi.h>


#include "tester_helper.h"


// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RANDOM ALLOCATION HELPER

double * ralloc( int sz ){
  double *X = (double *) malloc( sz *sizeof(double) );
  for (int i=0;i<sz;i++)
    X[i] = ( (double) (rand()) ) / (double) RAND_MAX;
  return X;
}


// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MPI TESTER MAIN FUNCTION

int testMPI( int    const n,
             int    const d,
             int    const k,
             int    const ap ){

  int p, id;                    // MPI # processess and PID
  MPI_Status Stat;              // MPI status
  int dst, rcv, tag;            // MPI destination, receive, tag

  int isValid = 0;              // return value

  MPI_Comm_rank(MPI_COMM_WORLD, &id); // Task ID
  MPI_Comm_size(MPI_COMM_WORLD, &p);  // # tasks


  // allocate corpus for each process
  double * const corpus = (double * ) malloc( n*d * sizeof(double) );

  if (id == 0) {                //============================== MASTER

    // ---------- Initialize data to begin with
    double const * const corpusAll = ralloc( n*d*p );


    // ---------- Break to subprocesses
    for (int ip = 0; ip < p; ip++){

      for (int i=0; i<n; i++)
        for (int j=0; j<d; j++)
          if (ap == COLMAJOR)
            corpus_cm(i,j) = corpusAll_cm(i+ip*n,j);
          else
            corpus_rm(i,j) = corpusAll_rm(i+ip*n,j);

      if (ip == p-1)            // last chunk is mine
        break;

      // which process to send? what tag?
      dst = ip+1;
      tag = 1;

      // send to correct process
      MPI_Send(corpus, n*d, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD);

    } // for (ip)


    // ---------- Run distributed kNN
    knnresult const knnres = distrAllkNN( corpus, n, d, k);


    // ---------- Prepare global kNN result object
    knnresult knnresall;
    knnresall.nidx  = (int *)   malloc( n*p*k*sizeof(int)    );
    knnresall.ndist = (double *)malloc( n*p*k*sizeof(double) );
    knnresall.m = n*p;
    knnresall.k = k;


    // ---------- Put my results to correct spot
    for (int j = 0; j < k; j++)
      for (int i = 0; i < n; i++){
        if (ap == COLMAJOR){
          knnresallnidx_cm(i+(p-1)*n,j)  = knnresnidx_cm(i,j);
          knnresallndist_cm(i+(p-1)*n,j) = knnresndist_cm(i,j);
        }else{
          knnresallnidx_rm(i+(p-1)*n,j)  = knnresnidx_rm(i,j);
          knnresallndist_rm(i+(p-1)*n,j) = knnresndist_rm(i,j);
        }
    }


    // ---------- Gather results back
    for (int ip = 0; ip < p-1; ip++){

      rcv = ip+1;
      tag = 1;

      MPI_Recv( knnres.nidx, n*k, MPI_INT, rcv, tag, MPI_COMM_WORLD, &Stat);
      MPI_Recv( knnres.ndist, n*k, MPI_DOUBLE, rcv, tag, MPI_COMM_WORLD, &Stat);

      for (int j = 0; j < k; j++)
        for (int i = 0; i < n; i++){
          if (ap == COLMAJOR){
            knnresallnidx_cm(i+ip*n,j)  = knnresnidx_cm(i,j);
            knnresallndist_cm(i+ip*n,j) = knnresndist_cm(i,j);
          }else{
            knnresallnidx_rm(i+ip*n,j)  = knnresnidx_rm(i,j);
            knnresallndist_rm(i+ip*n,j) = knnresndist_rm(i,j);
          }
        }

    }

    // printf ("\n Matrix DISTANCE: \n");
    // for (int i=0; i<n*p; i++) {
    //   for (int j=0; j<k; j++) {
    //     printf ("%lf    ", *(knnresall.ndist+i*k+j));
    //   }
    //   printf ("\n");
    // }
    //
    // printf ("\n Matrix INDECES: \n");
    // for (int i=0; i<n*p; i++) {
    //   for (int j=0; j<k; j++) {
    //     printf ("%10.2d", *(knnresall.nidx+i*k+j));
    //   }
    //   printf ("\n");
    // }

    // ---------- Validate results
    isValid = validateResult( knnresall, corpusAll, corpusAll,
                              n*p, n*p, d, k, ap );

  } else {                      //============================== SLAVE

    // ---------- Get data from MASTER
    rcv = 0;
    tag = 1;

    MPI_Recv(corpus, n*d, MPI_DOUBLE, rcv, tag, MPI_COMM_WORLD, &Stat);


    // ---------- Run distributed kNN
    knnresult const knnres = distrAllkNN( corpus, n, d, k);


    // ---------- Send data back to MASTER
    dst = 0;
    tag = 1;

    MPI_Send(knnres.nidx, n*k, MPI_INT, dst, tag, MPI_COMM_WORLD);
    MPI_Send(knnres.ndist, n*k, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD);

  }


  // ~~~~~~~~~~~~~~~~~~~~ Deallocate memory
  free( corpus );


  // ~~~~~~~~~~~~~~~~~~~~ Return wheter validations passed or not
  return isValid;

}


int main(int argc, char *argv[])
{

  MPI_Init(&argc, &argv);       // initialize MPI

  int id;                       // PID
  int n=500;                   // # corpus elements per process
  int d=2;                     // # dimensions
  int k=10;                     // # neighbors

  MPI_Comm_rank(MPI_COMM_WORLD, &id); // Task ID


  // ============================== RUN EXPERIMENTS

  //int isValidC = testMPI( n, d, k, COLMAJOR );
  int isValidC =0;
  int isValidR = testMPI( n, d, k, ROWMAJOR );


  // ============================== ONLY MASTER OUTPUTS

  if (id == 0) {                // ..... MASTER gets result
    printf("Tester validation: %s NEIGHBORS\n",
           STR_CORRECT_WRONG[isValidC||isValidR]);

  }

  MPI_Finalize();               // clean-up

  return 0;

}
