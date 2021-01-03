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

__global__
void copyCompress(
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

  if(place > inputSize){
    for(int in = 0; in < blockSize; ++in)
      outArr[outplace+in]  = 0;
    sizeArrGpu[caseId]     = 0;
    checksumArrGpu[caseId] = 0;
  }
  else if((place + blockSize) > inputSize){
    int endIn = inputSize - place;
    for(int in = endIn; in < blockSize; ++in)
      outArr[outplace+in] = 0;
    for(int in = 0; in < endIn; ++in)
      outArr[outplace+in] = inArr[place+in];
    sizeArrGpu[caseId]     = endIn;
  }
  else{
    for(int in = 0; in < blockSize; ++in)
      outArr[outplace+in] = inArr[place+in];
    sizeArrGpu[caseId] = blockSize;
  }

  compTypeArrGpu[caseId] = MATT_CMPRES_NUM_NONE;
}

