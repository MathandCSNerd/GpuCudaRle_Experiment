/* Copyright 2020 Matthew Macallister
 *
 * This file is part of GpuCudaRle_Experiment.
 *
 * GpuCudaRle_Experiment is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GpuCudaRle_Experiment is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GpuCudaRle_Experiment.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or
 * combining it with NVIDIA Corporation's CUDA libraries from the
 * NVIDIA CUDA Toolkit (or a modified version of those libraries),
 * containing parts covered by the terms of NVIDIA CUDA Toolkit
 * EULA, the licensors of this Program grant you additional
 * permission to convey the resulting work.
 */


#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "outputHandler.h"

using namespace std;

#define MATT_THREADS
outputHandler::outputHandler(const std::string& fileStr,
  size_t arrSize, size_t blockSize){

  mysize = arrSize;
  myBlockSize = blockSize;
  writer = std::make_shared<fileWriter>(fileStr, mysize, blockSize);
  cpuBuf = writer->getBuf();
  outputThread = nullptr;

  sizeOfCompTypeArrs = GPU_CORES*sizeof(unsigned char);
  sizeOfSizeArrs     = GPU_CORES*sizeof(unsigned int);
  sizeOfChecksumArrs = GPU_CORES*CHECKSUM_SIZE;

  compTypeArrCpu = (unsigned char *)     malloc(sizeOfCompTypeArrs);
  sizeArrCpu     = (unsigned int *)      malloc(sizeOfSizeArrs    );
  checksumArrCpu = (unsigned long long *)malloc(sizeOfChecksumArrs);

  _mallocGpu();
}

outputHandler::~outputHandler(){
  cudaFree(gpuBuf);
  joinWrite();

  cudaFree(compTypeArrGpu);
  cudaFree(sizeArrGpu);
  cudaFree(checksumArrGpu);

  free(compTypeArrCpu);
  free(sizeArrCpu);
  free(checksumArrCpu);
}

char* outputHandler::getBuf(){
  return gpuBuf;
}

unsigned char* outputHandler::getCompTypeArr(){
  return compTypeArrGpu;
}

unsigned long long* outputHandler::getChecksumArr(){
  return checksumArrGpu;
}

unsigned int* outputHandler::getSizeArr(){
  return sizeArrGpu;
}

bool outputHandler::startWrite(int size){
  _copyFromGpu();
  //cerr << "writing to disk " << "\n";
  #ifdef MATT_THREADS
  outputThread = std::make_shared<std::thread>
    (&outputHandler::_doWrite, this, size);
  #else
  doWrite(size);
  #endif
  //cerr << "writen " << "\n";
  return true;
}

bool outputHandler::joinWrite(){
  #ifdef MATT_THREADS
  if(outputThread)
    outputThread->join();

  outputThread = nullptr;
  #endif
  return true;
}

bool outputHandler::_copyFromGpu(){
  cerr << "copying output arrays to system memory" << endl;

  auto result = cudaMemcpy(cpuBuf, gpuBuf,
    mysize, cudaMemcpyDeviceToHost );
  PrintGpuErr(result);

  result = cudaMemcpy(compTypeArrCpu, compTypeArrGpu,
    sizeOfCompTypeArrs, cudaMemcpyDeviceToHost );
  PrintGpuErr(result);

  result = cudaMemcpy(sizeArrCpu, sizeArrGpu,
    sizeOfSizeArrs, cudaMemcpyDeviceToHost );
  PrintGpuErr(result);

  result = cudaMemcpy(checksumArrCpu, checksumArrGpu,
    sizeOfChecksumArrs, cudaMemcpyDeviceToHost );
  PrintGpuErr(result);

  return result != 0;
}

bool outputHandler::_mallocGpu(){
  cerr << "allocating output arrays to gpu" << endl;

  auto result = cudaMalloc( &gpuBuf,    mysize);
  PrintGpuErr(result);

  result = (cudaMalloc(&compTypeArrGpu, sizeOfCompTypeArrs));
  PrintGpuErr(result);

  result = (cudaMalloc(&sizeArrGpu,     sizeOfSizeArrs    ));
  PrintGpuErr(result);

  result = (cudaMalloc(&checksumArrGpu, sizeOfChecksumArrs));
  PrintGpuErr(result);

  return result != 0;
}

bool outputHandler::_doWrite(int size){
  for(int i = 0; i < GPU_CORES && sizeArrCpu[i]; ++i){
    writer->writeChar(compTypeArrCpu[i]);
    if(compTypeArrCpu[i])
      writer->writeInt(sizeArrCpu[i]);
    auto retVal = writer->writePiece(i*myBlockSize, sizeArrCpu[i]);
    if (retVal == 0)
      return false;
    //cerr << "qwuitting at index: " << i << ' ' << sizeArrCpu[i] << endl;
  }
  return true;
}

