#include "hip/hip_runtime.h"
#include <cstdlib>
#include <cstring>
#include <atomic>

dim3 blockIdx(0,0,0);
dim3 threadIdx(0,0,0);
dim3 blockDim(1,1,1);

const char* hipGetErrorString(hipError_t err) { return ""; }
int hipMalloc(void** ptr, size_t size) { 
    *ptr = std::malloc(size); 
    return 0; 
}
int hipFree(void* ptr) { 
    std::free(ptr); 
    return 0; 
}
int hipMemcpy(void* dst, const void* src, size_t size, int kind) { 
    std::memcpy(dst, src, size); 
    return 0; 
}
int hipGetLastError() { return 0; }

int atomicAdd(unsigned int* address, unsigned int val) {
    std::atomic<unsigned int>* atomic_addr = reinterpret_cast<std::atomic<unsigned int>*>(address);
    return atomic_addr->fetch_add(val);
}
int atomicSub(unsigned int* address, unsigned int val) {
    std::atomic<unsigned int>* atomic_addr = reinterpret_cast<std::atomic<unsigned int>*>(address);
    return atomic_addr->fetch_sub(val);
}
