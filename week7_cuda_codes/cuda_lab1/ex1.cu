/*to compile: nvcc ex1.cu -o ex1
  to run: ./ex1
  output: prints out first 16 elements*/
// include files
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// kernel routine
__global__ void my_kernel(float *x) {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  x[tid] = (float) threadIdx.x;
}
// main code
int main(int argc, char **argv) {
  float *h_x, *d_x;
  int   nblocks, nthreads, nsize, n;
  // set number of blocks, and threads per block
  nblocks  = 2;
  nthreads = 128;
  nsize    = nblocks*nthreads ;
  // allocate memory for array
  h_x = (float *)malloc(nsize*sizeof(float));
  cudaMalloc((void **)&d_x, nsize*sizeof(float));
  // execute kernel
  my_kernel<<<nblocks,nthreads>>>(d_x);
  // copy back results and print them out
  cudaMemcpy(h_x,d_x,nsize*sizeof(float),cudaMemcpyDeviceToHost);
  for (n=0; n<16; n++) printf(" n,  x  =  %d  %f \n",n,h_x[n]);
  // free memory
  cudaFree(d_x);
  free(h_x);
  // CUDA exit -- needed to flush printf write buffer
  cudaDeviceReset();
  return 0;
}
