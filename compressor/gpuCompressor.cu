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
#include "ioOps/fileReader.h"
#include "ioOps/fileWriter.h"
#include "gpuConstants.h"
#include "gpuFuncs.h"
#include "compression/methods.h"
#include "ioOps/inputHandler.h"
#include "ioOps/outputHandler.h"

using namespace std;

size_t getFSize(const std::string &fname){
  struct stat fileStats;
  stat(fname.c_str(), &fileStats);
  return fileStats.st_size;
}

int main(int argc, char** argv){
  size_t loopCount = 0;
  size_t blockSize = PARSE_BLOCK_SIZE;
  size_t arrByteCount = IN_ARR_BYTE_COUNT;

  std::string inFilename = "-";
  std::string outFilename = "-";

  if(argc > 1)
    inFilename = std::string(argv[1],strlen(argv[1]));

  if(argc > 2)
    outFilename = std::string(argv[2],strlen(argv[2]));
  else if(inFilename != "-")
    outFilename = inFilename + ".rle";

  if(argc > 3){
    auto tmpInt = atoi(argv[3]);
    if (tmpInt != 0){
      blockSize = tmpInt;
    }
    arrByteCount = blockSize*GPU_CORES;
  }

  auto inFileSize(getFSize(inFilename));

  if(argc <= 3 && inFileSize && arrByteCount > inFileSize){
    blockSize = max(size_t(1024), (inFileSize / GPU_CORES)+1);
    arrByteCount = ((inFileSize/GPU_CORES)+1)*GPU_CORES;
  }

  std::cerr << "gigs used on ram/vram: " <<
    (double(arrByteCount*2)/1024/1024/1024) << std::endl;

  size_t fullLoopCount = inFileSize/arrByteCount + bool(inFileSize%arrByteCount);

  inputHandler  inputer  (inFilename,  arrByteCount);
  outputHandler outputer (outFilename,  arrByteCount,
    blockSize);

  char * inputArrGpu = (inputer.getBuf());
  char * outputArrGpu = (outputer.getBuf());
  unsigned long long *checksumArrGpu(outputer.getChecksumArr());
  unsigned int *sizeArrGpu(outputer.getSizeArr());
  unsigned char *compTypeArrGpu  = (outputer.getCompTypeArr());

  inputer.startRead();
  unsigned long tim = time(NULL);

  while(!inputer.isAtEof())
  {
    inputer.joinRead();
    inputer.startRead();

    if(!inputer.hasDataToProcess())
      break;

    cerr << "progress: " << (++loopCount) << "/" << fullLoopCount << endl;
    syncGpu();
    size_t bytesToRead = inputer.bytesToRead();
    mrleCompress<<<GPU_BLOCKS, GPU_THREADS>>>(bytesToRead, blockSize, outputArrGpu, inputArrGpu,
        sizeArrGpu, checksumArrGpu, compTypeArrGpu);
    syncGpu();

    outputer.joinWrite();
    outputer.startWrite();
  }

  outputer.joinWrite();

  printf("finished in %li seconds\n",time(NULL)-tim);

  return 0;
}

