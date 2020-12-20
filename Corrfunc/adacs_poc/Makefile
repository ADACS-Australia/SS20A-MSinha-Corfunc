CC:=g++
CLC:=nvcc
EXE:=simple-corr

GCC_FLAGS:=-O2 -g -Wall -Wextra
NVCC_FLAGS:=-O2 -lOpenCL

all: CL TOOLS MAIN
	$(CC) $(NVCC_FLAGS) computeCL.o tools.o main.o -o test.run

CL:
	$(CLC) $(NVCC_FLAGS) computeCL.cpp -c

TOOLS:
	$(CC) $(GCC_FLAGS) tools.cpp -c

MAIN:
	$(CC) $(GCC_FLAGS) main.cpp -c

clean:
	-rm *.o
	-rm test.run
