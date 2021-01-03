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
#include "gpuConstants.h"

using namespace std;

__device__
void mrlePiece(
    size_t inputSize,
    size_t blockSize,
    char *outArr,
    char* inArr,
    unsigned int *sizeArrGpu,
    unsigned long long *checksumArrGpu,
    unsigned char *compTypeArrGpu
    ){

  const int threadId = threadIdx.x;
  const int blockId = blockIdx.x;
  const int caseId = threadId+blockId*GPU_THREADS;

  int place = (caseId*blockSize);
  int outplace = (caseId*blockSize);

  bool fail = false;

  int outIndex = 0;
  int count = 0;
  bool line = false;
  char last = (inArr[place]+1)%255;
  char ch;
  int in;
  size_t sizeOfThisBlock = blockSize;
  if(place + blockSize > inputSize)
    sizeOfThisBlock = inputSize % blockSize;

  for(in = 0; in < sizeOfThisBlock && !fail; ++in){
    ch = inArr[place+in];
    if(ch == last && count < 254){
      count += 1;
      line = true;
    }
    else if(line){
      outArr[outplace+outIndex] = RLE_ESCAPE;
      outIndex += 1;
      if(count >= RLE_ESCAPE)
        outArr[outplace+outIndex] = count+1;
      else
        outArr[outplace+outIndex] = count;
      outIndex += 1;
      outArr[outplace+outIndex] = ch;
      outIndex += 1;
      if(ch == RLE_ESCAPE){
        outArr[outplace+outIndex] = ch;
        outIndex += 1;
      }
      count = 0;
      line = false;
    }
    else if(ch == RLE_ESCAPE){
      outArr[outplace+outIndex] = ch;
      outIndex += 1;
      outArr[outplace+outIndex] = ch;
      outIndex += 1;
    }
    else{
      outArr[outplace+outIndex] = ch;
      outIndex += 1;
    }
    last = ch;
    if((blockSize - outIndex) < 10){
      fail = true;
    }
  }

  if(line && !fail){
    outArr[outplace+outIndex] = RLE_ESCAPE;
    outIndex += 1;
    if(count >= RLE_ESCAPE)
      outArr[outplace+outIndex] = count+1;
    else
      outArr[outplace+outIndex] = count;
    outIndex += 1;
    count = 0;
    line = false;
  }

  sizeArrGpu[caseId] = outIndex;
  compTypeArrGpu[caseId] = MATT_CMPRES_NUM_RLE;

  //if the "compressed" size would be greater than
  //before, just don't compress
  if(fail){
    for(int in2 = 0; in2 < sizeOfThisBlock; ++in2)
      outArr[outplace+in2] = inArr[place+in2];
    sizeArrGpu[caseId] = sizeOfThisBlock;
    compTypeArrGpu[caseId] = MATT_CMPRES_NUM_NONE;
  }
}

__global__
void mrleCompress(
    size_t inputSize,
    size_t blockSize,
    char *outArr,
    char* inArr,
    unsigned int *sizeArrGpu,
    unsigned long long *checksumArrGpu,
    unsigned char *compTypeArrGpu
  ){

  const int threadId = threadIdx.x;
  const int blockId = blockIdx.x;
  const int caseId = threadId+blockId*GPU_THREADS;

  int place = (caseId*blockSize);
  int outplace = (caseId*blockSize);

  sizeArrGpu[caseId]     = 0;
  checksumArrGpu[caseId] = 0;

  if(place > inputSize){
    for(int in = 0; in < blockSize; ++in)
      outArr[outplace+in]  = 0;
    sizeArrGpu[caseId]     = 0;
    checksumArrGpu[caseId] = 0;
    compTypeArrGpu[caseId] = MATT_CMPRES_NUM_NONE;
  }
  else if((place + blockSize) > inputSize){
    mrlePiece(inputSize, blockSize, outArr, inArr, sizeArrGpu, checksumArrGpu, compTypeArrGpu);
  }
  else{
    mrlePiece(inputSize, blockSize, outArr, inArr, sizeArrGpu, checksumArrGpu, compTypeArrGpu);
  }

}

