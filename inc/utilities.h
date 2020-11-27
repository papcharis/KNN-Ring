#ifndef UTILITIES_H
#define UTILITIES_H


/**
 *Method swapElement()
 *Description:
 *    Swaps the elements of two arrays
 */
void swapElement(double **one, double  **two);


/**
 *Method SumRow()
 *Description:
 *  Calculates the sum of each column's points squared.
 * Return value:
 *    1-D array with every point expressing one column of the previous
 *    array. 
 */

double SumRow(double *array, int numOfColumns, int row);

/**
 *Method qselect()
 *Description:
 *  Using method Quick Select sorts the array and keeps the k smallest points and their indeces.
 */

void qselect(double *tArray,int *index, int len, int k);

/**
 *Method quicksort()
 *Description:
 *  Sorts from first to last points of an array using Quicksort method.
 */
void quicksort(double *array, int *idx, int first, int last);

/**
 *Method changeResult()
 *Description:
 *  Takes two knnResults (of two comparisons) and keeps the k smallest distances and the right indeces.
 *Return value:
 *  Returns a knnresult-type  result with the k nearest neighbors of the comparison and their indeces.
 */
knnresult changeResult(knnresult result,knnresult tempResult,int offset,int newOff);



#endif
