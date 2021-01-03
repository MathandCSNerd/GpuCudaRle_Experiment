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
#include <vector>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include "fileWriter.h"
#include "gpuConstants.h"

using namespace std;

std::string fileWriter::sizetToString(size_t value){
  std::string retVal = "";
  while(value){
    char app = (value % 10) + '0';
    std::string apps = "";
    apps += app;
    retVal = apps + retVal;
    value /= 10;
  }
  if (retVal.length() == 0)
    retVal = "0";
  return retVal+"\n";
}

fileWriter::fileWriter(const std::string& fileStr, size_t arrSize, size_t blockSize){
  fail = false;
  mysize = arrSize;
  writeSize = blockSize;
  buf = new char[mysize];
  filename = fileStr;
  if(fileStr == "-")
    fd = STDOUT_FILENO;
  else
    fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);

  //std::string tmpOutStr(sizetToString(blockSize));
  //write(fd, tmpOutStr.c_str(), tmpOutStr.length());
  std::string verOut = MATT_GC_VERSION_CODE+"\n";
  cerr << "writing version code: " << verOut << endl;
  write(fd, verOut.c_str(), verOut.length());
  writeInt(blockSize);
}

fileWriter::~fileWriter(){
  free(buf);
  close(fd);
}

size_t fileWriter::writeInt(size_t size){
  std::string tmpOutStr(sizetToString(size));
  //cerr << "writing: " << size << ' ' << tmpOutStr << ' ' << tmpOutStr.length() << endl;
  size_t count = write(fd, tmpOutStr.c_str(), tmpOutStr.length());
  //cerr << "wrote: " << count << " bytes out of " << tmpOutStr.length() << endl;
  //count = write(fd, tmpOutStr.c_str(), tmpOutStr.length());
  return count;
}

size_t fileWriter::writeChar(char mychar){
  size_t count;

  count = write(fd, &mychar, 1);

  if (count != 1)
    fail = true;

  return count;
}

size_t fileWriter::writeArr(int size){
  size_t count;

  if(size != -1)
    count = write(fd, buf, size);
  else{
    count = write(fd, buf, mysize);
  }
  if (count != mysize)
    fail = true;
  return count;
}

size_t fileWriter::writePiece(unsigned long start, int size){
  size_t count;

  if(size > writeSize){
    cerr << "error, corrupted block, exiting: " << start <<  ' ' << size << ' ' << writeSize<< endl;
    return 0;
  }

  count = write(fd, buf+(start), size);

  if (count != mysize)
    fail = true;
  return count;
}

char* fileWriter::getBuf(){
  return buf;
}

