#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define DIMx 5
#define DIMy 5

// input and mask are globals
// below these are for testing purposes
float input[DIMx][DIMy];
float output[DIMx][DIMy];
float mask1[3][3]; // averaging
float mask2[3][3]; // sharpening
float mask3[3][3]; // edging


// l and p are the rows and colums of the mask respectively
// i and j are the coordinates of input data point we want to mask for 
void maskingFunc(float *inputImg , float *outputImg, int rows , int cols , int i, int j, int maskDimx,float mask[3][3]);

// Define the files that are to be save and the reference images for validation
const char *imageFilename = "ref_rotated.pgm";
const char *refFilename   = "ref_rotated.pgm";

const char *sampleName = "simpleTexture";


//TESTING FUNCTIONS
void init();
void printOut(float *p , int y, int x);
void printMask(float mask[3][3]);

void Convolve(int argc , char **argv);

int main( int argc, char **argv ){
    init();
    printMask(mask1);
    printMask(mask2);
    printMask(mask3);

    Convolve(argc, argv);
    return 0;
}

// To be called seperately for each block to be masked
void maskingFunc(float *inputImg , float *outputImg, int rows , int cols , int i, int j, int maskDimx, float mask[3][3] ){
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
            f = mask[l][p];
            sum += (f*y) ;
        }
    }
    // if(i== 0 && j==0) printf("the sum is %f\n",sum);
    outputImg[ i*cols + j] = sum - outputImg[ i*cols + j] ;
}

void Convolve(int argc , char **argv){

    //load image from disk
    float *hData = NULL;
    unsigned int width , height;

    // int q = 512*511;
    char* imagePath = sdkFindFilePath(imageFilename, argv[0]);
    
    if(imagePath == NULL){
        printf("Unable to source image file: %s\n",imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);
    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\nand size is %d\n",imageFilename, width, height,size);
    // printf("%f\n",hData[q]);
    // printOut(hData, width, height);

    float *outputImg = (float *) malloc(size);

    for(int k = 0; k < width * height; k++){
        int i = k/width;
        int j = k%width;
        maskingFunc(hData, outputImg, width , height, i , j , 3, mask2);
    }
    // printf("%f\n",outputImg[q]);

    // outputfileName
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
    sdkSavePGM(outputFilename, outputImg, width, height);
    printf("Wrote '%s'\n", outputFilename);
}


// For testing purposes
// Initialises a 5 by 5 matrix to be masked and a mask
void init(){
    // for the mask

    // for the input
    for(int i = 0; i < 3; i++){
        for(int j = 0; j< 3; j++){
            mask1[i][j] = (float) 1/9; // average
            mask2[i][j] = -1; // sharpening
            mask3[i][j] = (j == 0 && i != 1)?-1:(j == 0 && i == 1)? -2:(j == 2 && i != 1)?1:(j == 2 && i == 1)? 2:0; // edge
        }
    }
    mask2[1][1] = 9;
    

}

// prints the output picture
void printOut(float *p , int width, int height){
    int rows = height;
    int cols = width;
    printf("rows = %d, cols = %d\n", rows, cols);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            printf("%f ", p[i*cols + j]);
        }
        printf("\n");
    }
    printf("\n");

}

// Prints a mask
void printMask(float mask[3][3]){
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            printf("%f ",mask[i][j]);
        }
        printf("\n");
    }
    printf("\n");

}