#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "../inc/common/book.h"


#define maskDimx 3

// input and mask are globals
// below these are for testing purposes
float mask1[maskDimx*maskDimx]; // averaging
float mask2[maskDimx*maskDimx]; // sharpening
float mask3[maskDimx*maskDimx]; // edging


// l and p are the rows and colums of the mask respectively
// i and j are the coordinates of input data point we want to mask for 
void maskingFunc(float *inputImg , float *outputImg, int rows , int cols , int i, int j, float mask[maskDimx*maskDimx]);

// Define the files that are to be save and the reference images for validation
const char *imageFilename = "lena_bw.pgm";
const char *refFilename   = "ref_rotated.pgm";

const char *sampleName = "simpleTexture";
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


__global__ void globalConvolve(float *inIMG, float *outIMG, int *gParams ){
    // the 

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

int main( int argc, char **argv ){
    init();
    printMask(mask1);
    printMask(mask2);
    printMask(mask3);

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

    for(int k = 0; k < width * height; k++){
        int i = k/width;
        int j = k - (i*width);
        maskingFunc(inputImg, outputImg, width , height, i , j , mask);
    }
    
    writeFile(outputImg, "_Serial", imagePath);
    

    /*
        Here Begins the code for GLOBAL AND CONSTANT MEMORY
        We create storage mechanisms and get rid of them again upon usage
        No storage mechanism used here will be used for any other code sections
    */
    float *gInputImg = 0 ;
    float *gOut  = 0;
    float out[size];
    // out = (float*) malloc(size);
    int params[3] = {(int)width , (int)height,(int)maskDimx};
    int *gParams;
    
    checkCudaErrors(cudaMalloc((void**) &gOut , size));
    checkCudaErrors(cudaMalloc((void**) &gParams , 3*sizeof(int)));

    checkCudaErrors(cudaMalloc((void**) &gInputImg , size));
    checkCudaErrors(cudaMemcpy(gInputImg, inputImg , size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(gParams, params , 3*sizeof(int), cudaMemcpyHostToDevice));


    dim3 grids(8,8);
    dim3 threads(64,64);

    globalConvolve<<<threads,grids>>>(gInputImg, gOut,  gParams);
    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(out, gOut, size, cudaMemcpyDeviceToHost));
    cudaFree(gInputImg);
    cudaFree(gOut);
    writeFile(out , "_Global" , imagePath);
    
    

}

void writeFile(float *out , char* name , char* imagePath){
    char outputFilenames[1024];
    strcpy(outputFilenames, imagePath);
    strcpy(outputFilenames + strlen(imagePath) - 4,name);
    strcat(outputFilenames ,"_out.pgm");

    sdkSavePGM(outputFilenames, out, width, height);
    printf("Wrote '%s'\n", outputFilenames);
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