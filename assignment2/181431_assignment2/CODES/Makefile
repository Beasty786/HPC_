INC="../inc"
INCFLAGS=-I$(INC)
GLUTFLAGS=-lglut -lGL
OMPFLAG=-fopenmp
CC=gcc
NVCC=nvcc

all: simpleTexture
simpleTexture: Convolution.cu
	$(NVCC) $(INCFLAGS) Convolution.cu -o conv
clean:
	rm conv
