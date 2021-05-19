#include "../../common/helper_cuda.h"
#include "../../common/book.h"
#include <stdio.h>
#include <iostream>

__global__ void add(int a , int b, int* c) {
    *c = a + b;
}


int main(void){
    

    cudaDeviceProp prop;

    int count;
    HANDLE_ERROR(cudaGetDeviceCount(&count));

    for(int i = 0; i < count; i++){
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        printf("    --- General information for device %d ---\n",i);
        printf("Name:  %s\n", prop.name );
        printf("Compute capability: %d.%d \n" , prop.major , prop.minor);
        printf("Clock rate:  %d\n" , prop.clockRate);
        printf("Device copy overlap:   ");
        if(prop.deviceOverlap)
            printf("Enabled\n");
        else
            printf("Disabled\n");

        printf("Kernel execition timeout:   ");
        if(prop.kernelExecTimeoutEnabled)
            printf("Enabled\n");
        else    
            printf("Disabled\n");

        printf("   --- Memory Information for device %d ---\n",i);
        printf("Total global memory: %ld\n", prop.totalGlobalMem/1000000);
        printf("Total constant memory: %ld\n", prop.totalConstMem/1000000);
        printf("Max mem pitch: %ld\n", prop.memPitch);
        printf("Texture Alignment: %ld\n", prop.textureAlignment);
        printf("   --- MP Information for device %d ---\n",i);
        printf("Multiprocessor count:  %d\n", prop.multiProcessorCount);
        printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
        printf("Registers per mp: %d\n", prop.regsPerBlock);
        printf("Threads in warp:  %d\n", prop.warpSize);
        printf("Max threads per block:  %d\n" , prop.maxThreadsPerBlock);
        printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0] ,prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n\n", prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);

    }

    return 0;
}