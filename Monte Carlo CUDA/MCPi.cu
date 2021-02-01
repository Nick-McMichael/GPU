/* Program to compute Pi using Monte Carlo methods */

//I apologise for the lack of timings and other detail that I know you would submitted with this code however I've had some trouble getting nsight to work
//so much of this is optimised based on my own timings and understandings hwoever I feel it is as well optimised as I can get it without knowing further detail on the processes
//included and their individual timings such as the curand number genetration and the memcpys.

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include <time.h>
#include <curand_kernel.h>
#define SEED 35791246
//Please note that Block_Size and Thread_Size are predefined here due to a lack of creativitity and simply because these numbers seemed to perform best,
//alternatively making the Block_Size = 10 (The number of SMs in my GPU) and the Thread_Size = 128 (number of cuda cores per SM) works relatively well too but could produce
//very high Trials_Per_Thread which is defined later as the niter/Block_Size/Thread_Size, again this is an uncreative solution and it can be slightly inaccurate since I'm rounding
//it off to an int but it seems to largely work. It also performs much better when given iteration values higher than the 100 000 000 limit set in the prac.
#define Block_Size 128
#define Thread_Size 256
#define PI 3.1415926535

__global__ void Monte_Carlo_GPU(float *estimate, curandState *states, int Trials_Per_Thread) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	float x, y;

	curand_init(SEED, tid, 0, &states[tid]);  // 	Initialize curand


	for(int i = 0; i < Trials_Per_Thread; i++) {
		x = curand_uniform (&states[tid]);
		y = curand_uniform (&states[tid]);
		points_in_circle += (x*x + y*y <= 1.0f); // add to count if x, y falls within the circle
	}
	estimate[tid] = 4.0f * points_in_circle / (float) Trials_Per_Thread; // return estimate of pi
}

double Monte_Carlo_CPU(int niter) {
	//A word for word copy of the original algorithm.
	double x, y;
	srand(SEED);
	double z;
	double pi;
	int count=0;
	for (int i=0; i < niter; i++) {
      x = (double)rand()/RAND_MAX;
      y = (double)rand()/RAND_MAX;
      z = x*x+y*y;
      if (z<=1) count++;
      }
	pi=(double)count/niter*4;
	return pi;
	
}


int main(int argc, char* argv)
{
	int niter=0;
	printf("Enter the number of iterations used to estimate pi: ");
	scanf("%d",&niter);
	int Trials_Per_Thread = niter/Block_Size/Thread_Size; //It's not the best solution but I didnt know how to divide it without complicating things unnecessarily 
	
	clock_t start, stop;
	float host[Block_Size * Thread_Size];
	float *dev;
	curandState *devStates;
	
	start = clock();

	cudaMalloc((void **) &dev, Block_Size * Thread_Size * sizeof(float)); // allocate device memory for counts
	
	cudaMalloc( (void **)&devStates, Thread_Size * Block_Size * sizeof(curandState) ); //allocate device memory for curand states

	Monte_Carlo_GPU<<<Block_Size, Thread_Size>>>(dev, devStates, Trials_Per_Thread);

	cudaMemcpy(host, dev, Block_Size * Thread_Size * sizeof(float), cudaMemcpyDeviceToHost); 
	float pi_gpu;
	for(int i = 0; i < Block_Size * Thread_Size; i++) {
		pi_gpu += host[i];
	}

	pi_gpu /= (Block_Size * Thread_Size);

	stop = clock();

	printf("GPU pi calculated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);

	start = clock();
	float pi_cpu = Monte_Carlo_CPU(niter);
	stop = clock();
	printf("CPU pi calculated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);

	printf("CUDA estimate of PI = %f [error of %f]\n", pi_gpu, pi_gpu - PI);
	printf("CPU estimate of PI = %f [error of %f]\n", pi_cpu, pi_cpu - PI);
	
	return 0;
	
}


//Original Code as a reference, pretty much used as is for the serial function
/*
int main(int argc, char* argv)
{
   int niter=0;
   double x,y;
   int i,count=0;  //# of points in the 1st quadrant of unit circle
   double z;
   double pi;

   printf("Enter the number of iterations used to estimate pi: ");
   scanf("%d",&niter);

   // initialize random numbers 
   srand(SEED);
   count=0;
   for ( i=0; i<niter; i++) {
      x = (double)rand()/RAND_MAX;
      y = (double)rand()/RAND_MAX;
      z = x*x+y*y;
      if (z<=1) count++;
      }
   pi=(double)count/niter*4;
   printf("# of trials= %d , estimate of pi is %g \n",niter,pi);
   return 0;
}
*/