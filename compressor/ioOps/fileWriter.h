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


#ifndef MATT_FILE_WRITER
#define MATT_FILE_WRITER

#include <iostream>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

class fileWriter{
  public:
  fileWriter(const std::string& fileStr, size_t arrSize, size_t blockSize);
  ~fileWriter();
  size_t writeChar(char mychar);
  size_t writeArr(int size = -1);
  size_t writePiece(unsigned long start, int size);
  char* getBuf();

  std::string sizetToString(size_t value);
  size_t writeInt(size_t size);

  private:
  bool fail;
  size_t writeSize;
  size_t mysize;
  char * buf;
  std::string filename;
  int fd;
};

#endif
