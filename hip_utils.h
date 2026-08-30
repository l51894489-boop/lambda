#ifndef HIP_UTILS_H
#define HIP_UTILS_H

#ifdef USE_NVCC
  #include <cuda_runtime.h>
  #define hipMalloc cudaMalloc
  #define hipMemcpy cudaMemcpy
  #define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
  #define hipMemcpyHostToDevice cudaMemcpyHostToDevice
  #define hipSuccess cudaSuccess
  #define hipError_t cudaError_t
  #define hipGetLastError cudaGetLastError
  #define hipGetErrorString cudaGetErrorString
#else
  #include <hip/hip_runtime.h>
#endif
#include <iostream>
#include <cstdlib>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

#endif // HIP_UTILS_H
