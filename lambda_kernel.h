#ifndef LAMBDA_KERNEL_H
#define LAMBDA_KERNEL_H

#include "secp256k1.h"
#include "hip_utils.h"
#include "hip_utils.h"

struct DPResult {
    uint64_t x[4];
    uint256_t a;
    uint256_t b;
    uint32_t walk_id;
};

// Initializes the device memory and launches the kernel.
struct StepLocal { 
    ECPointJacobian point; 
    uint256_t a; 
    uint256_t b; 
};

struct DeviceWalkState {
    uint256_t a, b;
    ECPointJacobian R;
    uint32_t walk_id;
    uint64_t prng_state[4];
};

void launch_lambda_kernel(
    DeviceWalkState* d_walkers,
    DPResult* d_dp_buffer,
    uint32_t* d_dp_count,
    const StepLocal* d_localStepTable,
    uint32_t N_STEPS,
    int DP_BITS,
    int local_count,
    unsigned long long* iters,
    int key_range
);

#endif // LAMBDA_KERNEL_H
