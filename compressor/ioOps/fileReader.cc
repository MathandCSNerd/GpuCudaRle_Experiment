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
#include "fileReader.h"

using namespace std;

fileReader::fileReader(const std::string& fileStr, size_t blockSize){
  fail = false;
  mysize = blockSize;
  buf = new char[mysize];
  filename = fileStr;
  if(fileStr == "-")
    ;
  else
    freopen(filename.c_str(), "r", stdin);
}

fileReader::~fileReader(){
  free(buf);
}

size_t fileReader::readBlock(){
  //cerr << "reading " << mysize << " bytes" << endl;
  cin.read(buf, mysize);
  auto count = cin.gcount();
  if (count != mysize){
    fail = true;
    //cerr << "count off, reached eof " << count << ' ' << mysize << endl;
  }
  if (count == -1)
    count = 0;
  //if(!count)
  //  fail = true;
  return count;
}

char* fileReader::getBuf(){
  return buf;
}

bool fileReader::isAtEof(){
  return fail;
}

