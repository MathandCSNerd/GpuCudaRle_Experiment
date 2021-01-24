# GpuCudaRle_Experiment

This is experimental CUDA software I wrote for the purposes of run-length-encoding data on a gpu.

The primary purpose of this software was to compress multi-gigabyte (and sometimes multi-terabyte) data I had which also had very long runs of the same byte. Also importantly: the impact on the cpu had to be minimal because I would also be stressing the cpu with some other tasks at the same time.

I had originally planned to add some other types of compression (gzip for example), but I've decided to transition that code to OpenCL and abandon this project.

Lastly: Since it was only intended as a tool for my personal use, I made a few interface/design decisions which may seem odd. For instance: I set up the constants to work efficiently with my gpu. It would still get decent performance on any modern gpu provided it has at least ~3GB of VRAM, but this certainly does limit its usefulness for others in its current state.


# Building

To build this it requires a version of the nvidia cuda toolkit compatible with the nvidia GTX-1060. As well GNU Make.

To make, simply invoke make from the root directory of the repo.


# Usage
## Encoding

gpuCompressor [INPUT_FILE] [OUTPUT_FILE]

gpuCompressor is the program which encodes data. It takes two positional arguments, one for the input file and another for the output.

If no arguments are given, it reads raw data from stdin and writes encoded data to stdout.

If a single argument is given, it reads from INPUT_FILE and writes a new file named INPUT_FILE.rle.

If both arguments are given, it reads from INPUT_FILE and writes a new file named OUTPUT_FILE.


## Decoding

unRle [INPUT_FILE] [OUTPUT_FILE]

unRle is the program which decodes data. It takes two positional arguments, one for the input file and another for the output.

If no arguments are given, it reads raw data from stdin and writes encoded data to stdout.

If a single argument is given, it reads from INPUT_FILE and writes encoded data to stdout (Note: this is different from the encoding version since I'm not guaranteed a file extension).

If both arguments are given, it reads from INPUT_FILE and writes a new file named OUTPUT_FILE.
