#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "../inc/common/book.h"


#define maskDimx 3
#define tileWidth 8 // this will be used for the shared memory code

// input and mask are globals for the serial code
float mask1[maskDimx*maskDimx]; // averaging
float mask2[maskDimx*maskDimx]; // sharpening
float mask3[maskDimx*maskDimx]; // edging


// l and p are the rows and colums of the mask respectively
// i and j are the coordinates of input data point we want to mask for 
void maskingFunc(float *inputImg , float *outputImg, int rows , int cols , int i, int j, float mask[maskDimx*maskDimx]);

// Define the files that are to be save and the reference images for validation
const char *imageFilename = "lena_bw.pgm";

 //load image from disk
 float *inputImg = NULL;
 unsigned int width , height;


/*
    These functions are vital functions
*/ 
 void writeFile(float *out , char* name , char* imagePath);
 void Convolve(int argc , char **argv, float mask[maskDimx*maskDimx]);


 //TESTING FUNCTIONS
void init(); // this one initialises our masks
void printOut(float *p , int y, int x); // This one has the power to print out the values of our whole image
void printMask(float mask[maskDimx*maskDimx]); // This function prints out the mask of our fuctions


/*
    Create arrays of kernals for constant memory
    These here will be shared with the the Shared memory device kernel function
*/
__constant__ float edge[9] = {-1,0,1,-2,0,2,-1,0,1};
__constant__ float ave[9] = {1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9};
__constant__ float sharp[9] = {-1,-1,-1,-1,9,-1,-1,-1,-1};

// Global memory code for convolution
__global__ void globalConvolve(float *inIMG, float *outIMG, int *gParams ){
    int width = gParams[0];
    int height = gParams[1];
    int DIMx = gParams[2];
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    // unsigned int size = width*height;
 
        float sum = 0.0;
        int m = (DIMx - 1)/2; // This handles different size mask dimensions i.e 3, 5, 7 etc
        for(int l = 0; l < DIMx; l++){
            for(int p = 0; p < DIMx; p++){
                // y is the value in input
                // f is the value of the mask at given indices
                float y, f;
                y = (i-m+l) < 0 ? 0 : (j-m+p) < 0 ? 0 : (i-m+l)> (height-1) ? 0 : (j-m+p) > (width-1)? 0: inIMG[(i-m+l)*width + (j-m+p)];
                f = sharp[l*DIMx + p];
                sum += (f*y) ;
            }
        }
        outIMG[ i*width + j] = sum ;
    
}

// Shared memory code for convolution
__global__ void sharedConvolve(float *inputImg, float *outputImg, int *gParams){
    
    __shared__ float sharedMem[tileWidth][tileWidth];
    
    int width = gParams[0];
    int height = gParams[1];
    int DIMx = gParams[2];
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    
    float sum = 0;
    int m = (DIMx - 1)/2; // This handles different size mask dimensions i.e 3, 5, 7 etc

    sharedMem[threadIdx.x][threadIdx.y] = inputImg[i*height + j];
    __syncthreads();
    for(int p=0;p<DIMx;++p){
        for(int l=0;l<DIMx;++l){
            if((i-m+p)<0){
                sum = sum + 0; 
            }
            else if((j-m+l)>=width){
                sum = sum + 0;
            }
            else if((i-m+p)>=height){
                sum = sum + 0;
            }
            else if((j-m+l)<0){
                sum = sum + 0;
            }
            else if((threadIdx.x-m+l)<0){
                sum = sum + inputImg[(i-m+p)*width+(j-m+l)]*sharp[p*DIMx+l];
            }
            else if((threadIdx.x-m+l)>=tileWidth){
                sum = sum + inputImg[(i-m+p)*width+(j-m+l)]*sharp[p*DIMx+l];
            }
            else if((threadIdx.y-m+p)<0){
                sum = sum + inputImg[(i-m+p)*width+(j-m+l)]*sharp[p*DIMx+l];
            }
            else if((threadIdx.y-m+p)>=tileWidth){
                sum = sum + inputImg[(i-m+p)*width+(j-m+l)]*sharp[p*DIMx+l];
            }
            else{
                sum = sum + sharedMem[(threadIdx.x-m+l)][(threadIdx.y-m+p)]*sharp[p*DIMx+l];
            }
        }
    }
    outputImg[i*height+j] = sum;
}




/*
    Code for Texture memory below
*/
texture<float, 2, cudaReadModeElementType> tex;
texture<float,2,cudaReadModeElementType> tex_sharp;
texture<float,2,cudaReadModeElementType> tex_edge;
texture<float,2,cudaReadModeElementType> tex_av;


__global__ void texConvolve(float *outputImg,int *gParams){
    int width = gParams[0];
    int height = gParams[1];
    int DIMx = gParams[2];

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int m = (DIMx - 1)/2; // This handles different size mask dimensions i.e 3, 5, 7 etc

    float sum = 0;
    for(int p=0;p<DIMx;++p){
        for(int l=0;l<DIMx;++l){
            if((i-m+p)<0){
                sum = sum + 0; 
            }
            else if((j-m+l) >= height){
                sum = sum + 0;
            }
            else if((i-m+p) >= width){
                sum = sum + 0;
            }
            else if((j-m+l)<0){
                sum = sum + 0;
            }
            else{
                sum += tex2D(tex,j-m+l , i-m+p)*tex2D(tex_edge,l,p);
            }
        }
    }
    outputImg[i*height+j] = sum;

}



int main( int argc, char **argv ){
    init();

    Convolve(argc, argv, mask2);

    cudaDeviceReset();
    return 0;
}

// To be called seperately for each block to be masked
void maskingFunc(float *inputImg , float *outputImg, int rows , int cols , int i, int j, float mask[maskDimx*maskDimx] ){
    int L = maskDimx; // L is the mask's x-axis dimension 
    int P = maskDimx; // P is the mask's y-axis dimension
    float sum = 0.0;
    int m = (maskDimx - 1)/2; // This handles different size mask dimensions i.e 3, 5, 7 etc
    for(int l = 0; l < L; l++){
        for(int p = 0; p < P; p++){
            // y is the value in input
            // f is the value of the mask at given indices
            float y, f;
            y = (i-m+l) < 0 ? 0 : (j-m+p) < 0 ? 0 : (i-m+l)> (rows-1) ? 0 : (j-m+p) > (cols-1)? 0: inputImg[(i-m+l)*cols + (j-m+p)];
            f = mask[l*maskDimx + p];
            sum += (f*y) ;
        }
    }
    // if(i== 0 && j==0) printf("the sum is %f\n",sum);
    outputImg[ i*cols + j] = sum ;
}

void Convolve(int argc, char **argv, float mask[maskDimx*maskDimx]){
    // Get the image path for for the image file name
    char* imagePath = sdkFindFilePath(imageFilename, argv[0]);
    
    if(imagePath == NULL){
        printf("Unable to source image file: %s\n",imageFilename);
        exit(EXIT_FAILURE);
    }
    // Load the image here
    sdkLoadPGM(imagePath, &inputImg, &width, &height);
    unsigned int size = width * height * sizeof(float);

    float *outputImg = (float *) malloc(size);

    StopWatchInterface *se_timer = NULL;
    sdkCreateTimer(&se_timer);
    sdkStartTimer(&se_timer);
    for(int k = 0; k < width * height; k++){
        int i = k/width;
        int j = k - (i*width);
        maskingFunc(inputImg, outputImg, width , height, i , j , mask);
    }
    sdkStopTimer(&se_timer);
    printf("Processing time for serial: %f (ms)\n", sdkGetTimerValue(&se_timer));
    printf("%.2f Mpixels/sec\n",(width *height / (sdkGetTimerValue(&se_timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&se_timer);
    
    writeFile(outputImg, "_Serial", imagePath);
    
//--------------------------------------------------------------------------------------------------------------------
    /*
        Here Begins the code for GLOBAL AND CONSTANT MEMORY
        We create storage mechanisms and get rid of them again upon usage
        No storage mechanism used here will be used for any other code sections
    */
    float *gInputImg = 0 ;
    float *gOut  = 0;
    float *out;
    out = (float*) malloc(size);
    int params[3] = {(int)width , (int)height,(int)maskDimx};
    int *gParams;
    
    checkCudaErrors(cudaMalloc((void**) &gOut , size));
    checkCudaErrors(cudaMalloc((void**) &gParams , 3*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &gInputImg , size));
    checkCudaErrors(cudaMemcpy(gInputImg, inputImg , size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(gParams, params , 3*sizeof(int), cudaMemcpyHostToDevice));


    dim3 grids(8,8);
    dim3 threads(64,64);

    StopWatchInterface *g_timer = NULL;
    sdkCreateTimer(&g_timer);
    sdkStartTimer(&g_timer);
    globalConvolve<<<threads,grids>>>(gInputImg, gOut,  gParams);
    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&g_timer);
    printf("Processing time for global: %f (ms)\n", sdkGetTimerValue(&g_timer));
    printf("%.2f Mpixels/sec\n",(width *height / (sdkGetTimerValue(&g_timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&g_timer);


    checkCudaErrors(cudaMemcpy(out, gOut, size, cudaMemcpyDeviceToHost));
    cudaFree(gInputImg);
    cudaFree(gOut);
    writeFile(out , "_Global" , imagePath);
    free(out);

    cudaDeviceReset(); // starting a new cuda session below

    //-------------------------------------------------------------------------------------------------------------------------------------------------------------------
    /*
        Here we start with the shared memory code
    */
    // dim3 grids(8,8);
    // dim3 threads(64,64);
    float *sInputImg = 0 ;
    float *sOut  = 0;
    float *out_s;
    out_s = (float*) malloc(size);
    int sparams[3] = {(int)width , (int)height,(int)maskDimx};
    int *sParams;
    
    checkCudaErrors(cudaMalloc((void**) &sOut , size));
    checkCudaErrors(cudaMalloc((void**) &sParams , 3*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &sInputImg , size));
    checkCudaErrors(cudaMemcpy(sInputImg, inputImg , size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(sParams, sparams , 3*sizeof(int), cudaMemcpyHostToDevice));

    StopWatchInterface *s_timer = NULL;
    sdkCreateTimer(&s_timer);
    sdkStartTimer(&s_timer);

    sharedConvolve<<<threads, grids>>>(sInputImg, sOut, sParams);

    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());

    sdkStopTimer(&s_timer);
    printf("Processing time for shared: %f (ms)\n", sdkGetTimerValue(&s_timer));
    printf("%.2f Mpixels/sec\n",(width *height / (sdkGetTimerValue(&s_timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&s_timer);

    checkCudaErrors(cudaMemcpy(out_s, sOut, size, cudaMemcpyDeviceToHost));
    writeFile(out_s, "_Shared",imagePath);
    cudaFree(sOut);
    cudaFree(sInputImg);
    free(out_s);
    // free(imagePath);

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------
    //convulution mask
    float *sharpening = (float*)malloc(sizeof(float)*3*3);
    float *edgeDectection = (float*)malloc(sizeof(float)*3*3);
    float *averaging = (float*)malloc(sizeof(float)*3*3);
   

    float sharp[9] = {-1,-1,-1,-1,9,-1,-1,-1,-1};
    float edge[9] = {-1,0,1,-2,0,2,-1,0,1};
    // float av[9] = {1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9};
    sharpening = &sharp[0];
    edgeDectection = &edge[0];
    // averaging=&av[0];
    for(int i=0;i<maskDimx*maskDimx;++i){
        averaging[i] = 1/9;
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc sharp_cd = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc edge_cd = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc av_cd = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuArray;
    cudaArray *sharp_cu;
    cudaArray *edge_cu;
    cudaArray *av_cu;
    checkCudaErrors(cudaMallocArray(&cuArray,&channelDesc,width,height));
    checkCudaErrors(cudaMallocArray(&sharp_cu,&sharp_cd,3,3));
    checkCudaErrors(cudaMallocArray(&edge_cu,&edge_cd,3,3));
    checkCudaErrors(cudaMallocArray(&av_cu,&av_cd,maskDimx,maskDimx));
    checkCudaErrors(cudaMemcpyToArray(cuArray,0,0,inputImg,size,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToArray(sharp_cu,0,0,sharpening,3*3*sizeof(float),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToArray(edge_cu,0,0,edgeDectection,3*3*sizeof(float),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToArray(av_cu,0,0,averaging,maskDimx*maskDimx*sizeof(float),cudaMemcpyHostToDevice));

    // Set texture parameters
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModeLinear;
    //tex.normalized = true;    // access with normalized texture coordinates

    tex_sharp.addressMode[0] = cudaAddressModeWrap;
    tex_sharp.addressMode[1] = cudaAddressModeWrap;
    tex_sharp.filterMode = cudaFilterModeLinear;
    //tex_sharp.normalized = true;    // access with normalized texture coordinates

    tex_av.addressMode[0] = cudaAddressModeWrap;
    tex_av.addressMode[1] = cudaAddressModeWrap;
    tex_av.filterMode = cudaFilterModeLinear;
    //tex_av.normalized = true;    // access with normalized texture coordinates

    tex_edge.addressMode[0] = cudaAddressModeWrap;
    tex_edge.addressMode[1] = cudaAddressModeWrap;
    tex_edge.filterMode = cudaFilterModeLinear;
    //tex_edge.normalized = true;    // access with normalized texture coordinates


    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(tex, cuArray, channelDesc));
    checkCudaErrors(cudaBindTextureToArray(tex_sharp, sharp_cu, sharp_cd));
    checkCudaErrors(cudaBindTextureToArray(tex_edge, edge_cu, edge_cd));
    checkCudaErrors(cudaBindTextureToArray(tex_av, av_cu, av_cd));

    //timing texture
    float *txData = NULL;
    checkCudaErrors(cudaMalloc((void **) &txData, size));
 
    StopWatchInterface *t_timer = NULL;
    sdkCreateTimer(&t_timer);
    sdkStartTimer(&t_timer);
 
    texConvolve<<<threads,grids,0>>>(txData,sParams);
     
    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());
 
    sdkStopTimer(&t_timer);
    printf("Processing time for texture: %f (ms)\n", sdkGetTimerValue(&t_timer));
    printf("%.2f Mpixels/sec\n",(width *height / (sdkGetTimerValue(&t_timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&t_timer);
 
    // Allocate mem for the result on host side
    float *tex_out = (float *) malloc(size);
    checkCudaErrors(cudaMemcpy(tex_out,txData,size,cudaMemcpyDeviceToHost));
    writeFile(tex_out, "_texMem",imagePath);

    // free(sharpening);
    // free(edgeDectection);
    // free(averaging);
    free(imagePath);
    cudaFree(sParams);

}

void writeFile(float *out , char* name , char* imagePath){
    char outputFilenames[1024];
    strcpy(outputFilenames, imagePath);
    strcpy(outputFilenames + strlen(imagePath) - 4,name);
    strcat(outputFilenames ,"_out.pgm");

    sdkSavePGM(outputFilenames, out, width, height);
    printf("Wrote '%s'\n\n", outputFilenames);
}

// For testing purposes
// Initialises a 5 by 5 matrix to be masked and a mask
void init(){
    // for the mask

    // for the input
    for(int i = 0; i < maskDimx; i++){
        for(int j = 0; j< maskDimx; j++){
            mask1[i*maskDimx + j] = (float) 1/(maskDimx*maskDimx); // average
            mask2[i*maskDimx + j] = -1; // sharpening
            mask3[i*maskDimx + j] = (j == 0 && i != 1)?-1:(j == 0 && i == 1)? -2:(j == 2 && i != 1)?1:(j == 2 && i == 1)? 2.0:0.0; // edge
        }
    }
    mask2[(maskDimx/2)*maskDimx + maskDimx/2] = 9;
    

}

// prints the output picture
void printOut(float *p , int width, int height){
    int rows = height;
    int cols = width;
    printf("rows = %d, cols = %d\n", rows, cols);
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++){
            printf("%f ", p[i*cols + j]);
        }
        printf("\n");
    }
    printf("\n");

}

// Prints a mask
void printMask(float mask[maskDimx*maskDimx]){
    for(int i = 0; i < maskDimx; i++){
        for(int j = 0; j < maskDimx; j++){
            printf("%f ",mask[i*maskDimx + j]);
        }
        printf("\n");
    }
    printf("\n");

}