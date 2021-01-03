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


#ifndef MATT_GPU_CONSTANTS
#define MATT_GPU_CONSTANTS

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>
using namespace std;

// Kernel function to add the elements of two arrays
const int GPU_THREADS = 128;
const int GPU_BLOCKS = 10;

const size_t GPU_CORES=GPU_THREADS*GPU_BLOCKS;
const size_t ONE_KILOBYTE=1024;                //byte count of a kb
const size_t ONE_MEGABYTE=1024*ONE_KILOBYTE;   //byte count of a mb

const size_t BYTE_DIVS=1;
const size_t PARSE_BLOCK_SIZE = ONE_MEGABYTE/BYTE_DIVS;
//const size_t PARSE_BLOCK_SIZE = ONE_KILOBYTE/BYTE_DIVS;
const size_t BYTES_TO_READ=GPU_CORES*PARSE_BLOCK_SIZE;

const size_t IN_ARR_BYTE_COUNT = BYTES_TO_READ*sizeof(char);
const size_t TOTAL_ARR_BYTE_COUNT = IN_ARR_BYTE_COUNT*2;

static unsigned int CHECKSUM_SIZE = sizeof(unsigned long long);

const char RLE_ESCAPE='\\';

const char MATT_CMPRES_NUM_NONE = 0;
const char MATT_CMPRES_NUM_RLE  = 1;

const std::string MATT_GC_VERSION_CODE = "MATT_GC-1";

#endif
