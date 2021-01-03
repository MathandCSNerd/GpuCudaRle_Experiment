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
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <string.h>
#include <memory>

using namespace std;

const size_t MAX_BUFSIZE=1024*1024;
const unsigned char RLE_ESCAPE='\\';

size_t getFSize(const std::string &fname){
  struct stat fileStats;
  stat(fname.c_str(), &fileStats);
  return fileStats.st_size;
}

shared_ptr<std::string> parseBlock(unsigned char* buf, size_t currBlockSize){
  auto retVal (std::make_shared<std::string>(""));
  for(size_t i = 0; i < currBlockSize; ++i){
    if(buf[i] == RLE_ESCAPE && buf[i+1] != RLE_ESCAPE){
      int count = buf[i+1];
      count+=1;
      if(buf[i+1] > RLE_ESCAPE)
        --count;
      while(--count > 0)
        *retVal += buf[i-1];
      ++i;
    }
    else if(buf[i] == RLE_ESCAPE && buf[i+1] == RLE_ESCAPE){
      *retVal += buf[i];
      ++i;
    }
    else{
      *retVal += buf[i];
    }
  }
  return retVal;
}

int main(int argc, char** argv){
  std::string inFilename = "-";
  std::string outFilename = "-";

  if(argc > 1)
    inFilename = std::string(argv[1],strlen(argv[1]));
  if(argc > 2)
    outFilename = std::string(argv[2],strlen(argv[2]));

  auto inFileSize(getFSize(inFilename));

  istream* infile;
  ifstream tmpinfile;
  if(inFilename == "-")
    infile = &cin;
  else{
    tmpinfile.open(inFilename);
    cerr << "openinig" << endl;
    if(!tmpinfile)
      cerr << "open failed?" << endl;
    infile = &tmpinfile;
  }

  ostream* outfile;
  ofstream tmpoutfile;
  if(outFilename == "-")
    outfile = &cout;
  else{
    tmpoutfile.open(outFilename);
    outfile = &tmpoutfile;
  }

  char dummychar;
  char compType;
  char tmpbuf[MAX_BUFSIZE];
  unsigned char buf[MAX_BUFSIZE];

  size_t maxBlockSize;
  size_t currBlockSize;
  size_t outCount(0);

  std::string verCode = "";
  getline(*infile, verCode);
  *infile >> maxBlockSize;
  infile->get(dummychar);

  cerr << "got here" << endl;

  while(infile->get(compType) && (!compType || *infile >> currBlockSize)){
    if(!compType)
      currBlockSize = maxBlockSize;
    else
      infile->get(dummychar);

    infile->read(tmpbuf, currBlockSize);
    currBlockSize = infile->gcount();

    memcpy(buf, tmpbuf, currBlockSize);

    outCount += currBlockSize;
    cerr << "progress: " << outCount << "/" << inFileSize << endl;

    if(compType == 0 /*|| currBlockSize == maxBlockSize || infile->peek() == EOF*/ ){
      outfile->write(tmpbuf, currBlockSize);
      //*outfile << std::string(tmpbuf, currBlockSize);
    }
    else{
      auto outVal (parseBlock(buf, currBlockSize));
      *outfile << *outVal;
    }
  }

  return 0;
}
