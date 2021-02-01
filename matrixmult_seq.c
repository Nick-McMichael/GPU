#include "stdio.h"
#include "stdlib.h"
#include <time.h>    // change this to use time.h for Microsoft
#include "math.h"

/*
	Run with the arguments: matrixmult_seq.exe 500 100
	Running on the CPU the program had a execution time of roughly 11s whilst running on the GPU I could acheive 0.7s.
	This speed-up is entirely gained in the matrixmult kernel as it had the only sizeable execution time, initiallly I used kernels but found I got fractionally better
	execution time if I handled the vector length manually and after some trial and error came to the conclusion the best speed-up was acheived with a vector size of 32 
	requiring 4 workers. I then noticed in the profiler that some time was wasted copying data in and out but I was unable to find a way to reduce this as any attempt to
	parallise the copyMatrix function generated slow-down and the same was true of attempting to use the update pragma so the data wouldnt go to the host and simply update on the 
	device, but this only created errors or further slow-down when i got it to work as the data set simply was large enough for it to be worth the additional overhead. 
	It can be seen in the profiler that compartively (on larger data sets when using less than 100 by 100 this no longer holds true) that the time taken for the copy is marginal
	when compared to the compute time for the MatrixMult function the same is true of the copymatrix function when comapred to the matrixmult.

*/

int main (int argc, char **argv);

void fillMatrix(int size, float **restrict A) {
   for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        A[i][j] = ((float)i);
      }
   }
}
float** MatrixMult(int size, float **restrict A, float **restrict B, float **restrict C) {
#pragma acc data copyin(A[0:size][0:size],B[0:size][0:size])
#pragma acc data copyout(C[0:size][0:size])
#pragma acc parallel vector_length(32)
#pragma acc loop gang worker
   for (int i = 0; i < size; ++i) {
#pragma acc loop vector
     for (int j = 0; j < size; ++j) {
       float tmp = 0.;
       for (int k = 0; k < size; ++k) {
          tmp += A[i][k] * B[k][j];
       }
       C[i][j] = tmp;
     }
   }
   return C;
}
float** MakeMatrix(int size, float **restrict arr) {
    int i;
    arr = (float **)malloc( sizeof(float *) * size);
    arr[0] = (float *)malloc( sizeof(float) * size * size);
    for (i=1; i<size; i++){
       arr[i] = (float *)(arr[i-1] + size);
    }
    return arr;
}
void showMatrix(int size, float **restrict arr) {
   int i, j;
   for (i=0; i<size; i++){
      for (j=0; j<size; j++){
         printf("arr[%d][%d]=%f \n",i,j,arr[i][j]);
      }
   }
}
void copyMatrix(float **restrict A, float **restrict B, int size){

   for (int i=0; i<size; ++i){
      for (int j=0; j<size; ++j){
         A[i][j] = B[i][j];
      }
   }
}
int main (int argc, char **argv) {
   int i, j, k;
   float **A, **B, **C;
     
   if (argc != 3) {
      fprintf(stderr,"Use: %s size nIter\n", argv[0]);
      return -1;
   }
   int size = atoi(argv[1]);
   int nIter = atoi(argv[2]);
    
   if (nIter <= 0) {
      fprintf(stderr,"%s: Invalid nIter (%d)\n", argv[0],nIter);
      return -1;
   }

   clock_t start_time, stop_time; // timers --- change for MS

    A = (float**)MakeMatrix(size, A);
   fillMatrix(size, A);
   B = (float**)MakeMatrix(size, B);
   fillMatrix(size, B);
   C = (float**)MakeMatrix(size, C);

   start_time = clock();  // Unix timer --- change for MS

   for (int i=0; i<nIter; i++) {
      C = MatrixMult(size, A, B, C);
      if (i%2==1) {
         copyMatrix(A, C, size); //multiply A by B and assign back to A on even iterations
      }
      else {
         copyMatrix(B, C, size); //multiply A by B and assign back to B on odd iterations
      }
   }
   //showMatrix(size, C);
   stop_time = clock();
   float diff = ((float)(stop_time - start_time) / CLOCKS_PER_SEC);  // timers --- change for MS
 
   printf("%s total runtime %8.5g\n", argv[0], diff);  
   free(A); free(B); free(C); 
   return 0;
}

