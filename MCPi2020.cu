////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo method to calculate PI
// Inefficient code using CURAND
// solution 2020
// execute using nvcc MCPi2020.cu -I /usr/local/cuda-10.0/samples/common/inc -lcurand
////////////////////////////////////////////////////////////////////////

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <cuda.h>
#include <curand.h>
#include <ctime>

using namespace std;


__global__ void calcPI (int n, float *d_x, float *d_y, int *d_count, int iter )
// Applies Monte Carlo method to compute PI 
{
   int i = (blockIdx.x*blockDim.x + threadIdx.x);
   int index = i * iter;  // as each processor does a number of iterations
   double x,y,z;
   int tempcount = d_count[i];
   if (i<n)   // to prevent excess processors from accessing out-of-bounds data
   { for(int j=0; j<iter; j++)   // iterations of Monte Carlo - based on parameter iter
      { x = d_x[index+j];   // optimise using only one array ....
        y = d_y[index+j]; 
        z = x*x+y*y;
        if (z<=1) tempcount++;   
      }
    d_count[i] = tempcount;
   }
}


int main (int argc, char*argv[] )
{		int iterations = 100; // number of iterations for MC alg per processor
		int n = 1000000; // total no. of iterations of MC // max storage = 1 500 000
		int block = 32; // block size
		int grid = n / block + 1;  // round up number of blocks

		int count=0; /* # of points in the 1st quadrant of unit circle */
		double pi;


 // initialise card

	findCudaDevice(argc, (const char**) argv);   // from helper_cuda.h 
 
  // initialise CUDA timing

	float milli;
	cudaEvent_t start, stop, starttot, stoptot;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	clock_t iStart, iEnd;
	iStart = clock();

	// allocate memory on host and device
		int *h_count = (int *)malloc(sizeof(int)*n);    // for processor sums
		float *d_x;  // memory for initial x random numbers
		checkCudaErrors(cudaMalloc((void**)&d_x,sizeof(float)*n*iterations));
		float *d_y;  // memory for initial y random numbers
		checkCudaErrors(cudaMalloc((void**)&d_y,sizeof(float)*n*iterations));
		int *d_count; // memory for intermediate results
		checkCudaErrors(cudaMalloc((void**)&d_count,sizeof(int)*n));
 		checkCudaErrors(cudaMemset(d_count,0, sizeof(int)*n));
 
	// random number generation
		cudaEventRecord(start);  // start timing

		curandGenerator_t gen;
		checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
		checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
		checkCudaErrors( curandGenerateUniform(gen, d_x, n*iterations) );
 		checkCudaErrors( curandGenerateUniform(gen, d_y, n*iterations) );
 
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milli, start, stop);  // time random generation

		printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n",
          milli, n/(0.001*milli));

	// execute kernel and time it

		cudaEventRecord(start); // start timing

		calcPI<<<grid,block>>>(n, d_x, d_y, d_count, iterations);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milli, start, stop);  // stop timing actual kernel execution

		printf("Square root kernel execution time (ms): %f \n",milli);

		checkCudaErrors(cudaDeviceSynchronize());  // flush print queues

		// copy back results
		checkCudaErrors( cudaMemcpy(h_count, d_count, sizeof(int)*n, cudaMemcpyDeviceToHost) );

		for (int k = 0; k < n; k++) // sum all counts
		 {  count += h_count[k];
		    // printf("count %i = %i \n", k, h_count[k]); 
		  }  

		pi=(double)count/(n*iterations)*4;

		iEnd = clock();

		printf("Total PI execution time (ms): %g \n", (double) ((iEnd-iStart) * 1000 / CLOCKS_PER_SEC)); 

		printf("# of trials= %d , estimate of pi is %g \n",n*iterations,pi);
	
	 // Tidy up library

		checkCudaErrors( curandDestroyGenerator(gen) );

	// Release memory and exit cleanly

		free(h_count);
		checkCudaErrors( cudaFree(d_count) );
		checkCudaErrors( cudaFree(d_x) );
		checkCudaErrors( cudaFree(d_y) );

	// CUDA exit 

		cudaDeviceReset();
}


