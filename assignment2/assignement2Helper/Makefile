INC="./inc"
INCFLAGS=-I$(INC)
GLUTFLAGS=-lglut -lGL
OMPFLAG=-fopenmp
CC=gcc
NVCC=nvcc

all: simpleTexture
simpleTexture: simpleTexture.cu
	$(NVCC) $(INCFLAGS) simpleTexture.cu -o simpleTexture
clean:
	rm simpleTexture
