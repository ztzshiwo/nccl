#pragma once
using cudaError_t = int;
constexpr int cudaSuccess = 0;
inline const char* cudaGetErrorString(int) { return "host CUDA stub"; }
inline int cudaGetDevice(int* device) { *device = 0; return 0; }
