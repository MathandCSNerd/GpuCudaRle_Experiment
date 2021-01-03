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
#include "inputHandler.h"

using namespace std;

#define MATT_THREADS

inputHandler::inputHandler(const std::string& fileStr, size_t blockSize){
  mysize = blockSize;
  reader = std::make_shared<fileReader>(fileStr, mysize);
  _mallocGpu();
  cpuBuf = reader->getBuf();
  bytesRead = 0;
  tempBytesRead = 0;
  eof = false;
  inputThread = nullptr;
}

inputHandler::~inputHandler(){
  cudaFree(gpuBuf);
  #ifdef MATT_THREADS
  inputThread->join();
  #endif
}

char* inputHandler::getBuf(){
  return gpuBuf;
}

bool inputHandler::isAtEof(){
  return eof;
}

bool inputHandler::startRead(){
  //std::cerr << "start" << "\n";
  #ifdef MATT_THREADS
  inputThread = std::make_shared<std::thread>(&inputHandler::_doRead, this);
  #else
  _doRead();
  #endif
  return true;
}

bool inputHandler::joinRead(){
  #ifdef MATT_THREADS
  inputThread->join();
  #else
  #endif
  bytesRead = tempBytesRead;
  eof |= reader->isAtEof();
  _copyToGpu();
  return true;
}

bool inputHandler::hasDataToProcess(){
  //cerr << "data to proc: " << bytesToRead() << ' ' << errno << endl;
  return bytesToRead() != 0;
}

int inputHandler::bytesToRead(){
  return bytesRead;
}

bool inputHandler::_copyToGpu(){
  cerr << "copying input array to gpu" << endl;

  auto result = cudaMemcpy( gpuBuf, cpuBuf, bytesRead, cudaMemcpyHostToDevice );
  PrintGpuErr(result);
  return result != 0;
}

bool inputHandler::_mallocGpu(){
  cerr << "allocating input array to gpu" << endl;

  auto result = cudaMalloc( &gpuBuf, mysize);
  PrintGpuErr(result);
  return result != 0;
}

bool inputHandler::_doRead(){
  tempBytesRead = reader->readBlock();
  return true;
}

