CC=nvcc
NVCC_FLAGS=-gencode arch=compute_60,code=sm_60
CXXFLAGS=
CXXFLAGS+= -I .

.SUFFIXES: .cu

.cc.o:
	$(CC) ${CXXFLAGS} $(CFLAGS) $< -c $@

.cu.o:
	$(CC) ${CXXFLAGS} $(CFLAGS) $(NVCC_FLAGS) $< -c $@

%.o: */%.cu
	$(CC) ${CXXFLAGS} $(CFLAGS) $(NVCC_FLAGS) $< -c $@

%.o: */%.cc
	$(CC) ${CXXFLAGS} $(CFLAGS) $(NVCC_FLAGS) $< -c $@

all: gpuCompressor

gpuCompressor: gpuCompressor.o gpuFuncs.o fileReader.o fileWriter.o inputHandler.o outputHandler.o copy.o rle.o

gpuCompressor.o:

gpuFuncs.o:

fileReader.o:

fileWriter.o:

inputHandler.o:

outputHandler.o:

copy.o:

rle.o:

run: all
	./gpuCompressor

clean:
	-rm gpuCompressor *.o
