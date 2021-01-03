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


#ifndef MATT_OUTPUT_HANDLER
#define MATT_OUTPUT_HANDLER

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <memory>
#include <thread>
#include "fileWriter.h"
#include "gpuConstants.h"
#include "gpuFuncs.h"

using namespace std;

class outputHandler{
  public:
  outputHandler(const std::string& fileStr, size_t arrSize, size_t blockSize);
  ~outputHandler();
  size_t writeBlock();
  char* getBuf();
  unsigned char* getCompTypeArr();
  unsigned long long* getChecksumArr();
  unsigned int* getSizeArr();
  bool startWrite(int size = -1);
  bool joinWrite();

  private:
  bool _copyFromGpu();
  bool _mallocGpu();
  bool _doWrite(int size = -1);

  size_t mysize;
  size_t myBlockSize;
  char * gpuBuf;
  char * cpuBuf;

  unsigned char* compTypeArrCpu;
  unsigned char* compTypeArrGpu;

  unsigned long long *checksumArrCpu;
  unsigned long long *checksumArrGpu;

  unsigned int *sizeArrCpu;
  unsigned int *sizeArrGpu;

  size_t sizeOfCompTypeArrs;
  size_t sizeOfSizeArrs;
  size_t sizeOfChecksumArrs;

  std::shared_ptr<fileWriter> writer;

  std::shared_ptr<std::thread> outputThread;
};

#endif
