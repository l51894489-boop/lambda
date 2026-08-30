#ifndef HIP_MOCK_H
#define HIP_MOCK_H
#include <cstddef>
#define __device__
#define __host__
#define __global__
#define hipSuccess 0
typedef int hipError_t;
const char* hipGetErrorString(hipError_t err);
int hipMalloc(void** ptr, size_t size);
int hipFree(void* ptr);
int hipMemcpy(void* dst, const void* src, size_t size, int kind);
int hipGetLastError();
#define hipMemcpyHostToDevice 1
#define hipMemcpyDeviceToHost 2
struct dim3 { int x, y, z; dim3(int x=1, int y=1, int z=1) : x(x), y(y), z(z) {} };
typedef struct dim3 dim3;
extern dim3 blockIdx;
extern dim3 threadIdx;
extern dim3 blockDim;
int atomicAdd(unsigned int* address, unsigned int val);
int atomicSub(unsigned int* address, unsigned int val);
#endif
