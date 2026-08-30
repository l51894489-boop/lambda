#ifndef HIP_UTILS_H
#define HIP_UTILS_H

#include <hip/hip_runtime.h>
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
