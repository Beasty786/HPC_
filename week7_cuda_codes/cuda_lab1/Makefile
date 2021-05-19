INC="../common"
INCFLAGS=-I$(INC)
GLUTFLAGS=-lglut -lGL
OMPFLAG=-fopenmp
CC=gcc
NVCC=nvcc

all: ex1 ex2 ex3 double vec_add julia_cpu
ex1: ex1.cu
	$(NVCC) ex1.cu -o ex1
ex2: ex2.cu
	$(NVCC) $(INCFLAGS) ex2.cu -o ex2
ex3: ex3.cu
	$(NVCC) $(INCFLAGS) ex3.cu -o ex3
double: double.cu
	$(NVCC) $(INCFLAGS) double.cu -o double
vec_add: vector_addition.cu
	$(NVCC) $(INCFLAGS) vector_addition.cu -o vec_add
julia_cpu: julia_cpu.cu
	$(NVCC) $(INCFLAGS) $(GLUTFLAGS) julia_cpu.cu -o julia_cpu
clean:
	rm ex1 ex2 ex3 double vec_add julia_cpu
