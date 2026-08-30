#include "lambda_kernel.h"

// MurmurHash3 avalanche mixer
__device__ inline uint32_t murmur3_mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return static_cast<uint32_t>(x);
}

__device__ inline uint32_t get_step_idx(const uint64_t* x, uint32_t N_STEPS) {
    uint64_t combined = x[0] ^ (x[1] << 1) ^ (x[2] << 2) ^ (x[3] << 3);
    return murmur3_mix(combined) % N_STEPS;
}

__device__ inline bool DP(const uint64_t* affine_x, int DP_BITS) {
    return (affine_x[0] & ((1ULL << DP_BITS) - 1)) == 0;
}

// Batch inversion using Montgomery trick (Fermat's Little Theorem for single inversion)
__device__ void batchJacobianToAffineDevice(
    ECPointAffine* aff_out,
    const ECPointJacobian* jac_in,
    int count,
    uint64_t* scratch_prefix,
    uint64_t* scratch_inv
) {
    if (count <= 0) return;
    
    uint64_t ONE_MONT[4] = { 0x00000001000003D1ULL, 0x0ULL, 0x0ULL, 0x0ULL };
    
    if (jacobianIsInfinity(&jac_in[0])) {
        for(int k=0; k<4; k++) scratch_prefix[0*4+k] = ONE_MONT[k];
    } else {
        for(int k=0; k<4; k++) scratch_prefix[0*4+k] = jac_in[0].Z[k];
    }

    for (int i = 1; i < count; i++) {
        if (jacobianIsInfinity(&jac_in[i])) {
            for(int k=0; k<4; k++) scratch_prefix[i*4+k] = scratch_prefix[(i-1)*4+k];
        } else {
            modMulMontP(&scratch_prefix[i*4], &scratch_prefix[(i-1)*4], jac_in[i].Z);
        }
    }

    uint64_t P_MINUS_2[4] = { 0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
    modExpMontP(&scratch_inv[(count-1)*4], &scratch_prefix[(count-1)*4], P_MINUS_2);

    for (int i = count - 1; i > 0; i--) {
        if (jacobianIsInfinity(&jac_in[i])) {
            for(int k=0; k<4; k++) scratch_inv[(i-1)*4+k] = scratch_inv[i*4+k];
        } else {
            modMulMontP(&scratch_inv[(i-1)*4], &scratch_inv[i*4], jac_in[i].Z);
            modMulMontP(&scratch_inv[i*4], &scratch_inv[i*4], &scratch_prefix[(i-1)*4]);
        }
    }

    for (int i = 0; i < count; i++) {
        if (jacobianIsInfinity(&jac_in[i])) {
            for(int k=0; k<4; k++) { aff_out[i].x[k] = 0; aff_out[i].y[k] = 0; }
            aff_out[i].infinity = 1;
        } else {
            uint64_t z_inv2[4], z_inv3[4];
            modMulMontP(z_inv2, &scratch_inv[i*4], &scratch_inv[i*4]);
            modMulMontP(z_inv3, z_inv2, &scratch_inv[i*4]);
            modMulMontP(aff_out[i].x, jac_in[i].X, z_inv2);
            modMulMontP(aff_out[i].y, jac_in[i].Y, z_inv3);
            aff_out[i].infinity = 0;
        }
    }
}

__global__ void lambda_walk_kernel(
    DeviceWalkState* d_walkers,
    DPResult* d_dp_buffer,
    uint32_t* d_dp_count,
    const StepLocal* d_localStepTable,
    uint32_t N_STEPS,
    int DP_BITS,
    int walkers_per_thread,
    int total_walkers,
    int key_range
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start_idx = tid * walkers_per_thread;
    if (start_idx >= total_walkers) return;

    int local_count = walkers_per_thread;
    if (start_idx + local_count > total_walkers) {
        local_count = total_walkers - start_idx;
    }

    double exp_steps = std::pow(2, key_range / 2.0);
    // Dynamic allocation could blow up registers, so we use a small fixed size limit (e.g. 16)
    const int MAX_W = 16;
    if (local_count > MAX_W) local_count = MAX_W;

    ECPointJacobian jac_batch[MAX_W];
    ECPointAffine aff_batch[MAX_W];
    uint64_t scratch_prefix[MAX_W * 4];
    uint64_t scratch_inv[MAX_W * 4];

    for (int i = 0; i < local_count; i++) {
        jac_batch[i] = d_walkers[start_idx + i].R;
    }

    batchJacobianToAffineDevice(aff_batch, jac_batch, local_count, scratch_prefix, scratch_inv);

    // Perform a chunk of steps (e.g. 256) per kernel invocation to amortize launch overhead
    for (int step = 0; step < 256; step++) {
        for (int i = 0; i < local_count; i++) {
            DeviceWalkState* w = &d_walkers[start_idx + i];
            
            uint32_t step_idx = get_step_idx(aff_batch[i].x, N_STEPS);
            bool negate = aff_batch[i].y[0] & 1;
            
            ECPointJacobian step_point = d_localStepTable[step_idx].point;
            if (negate) {
                uint64_t zero[4] = {0, 0, 0, 0};
                modSubP(step_point.Y, zero, step_point.Y);
            }
            
            pointAddJacobian(&w->R, &w->R, &step_point);
            
            if (negate) {
                scalarSub(w->a.limbs, w->a.limbs, d_localStepTable[step_idx].a.limbs);
            } else {
                scalarAdd(w->a.limbs, w->a.limbs, d_localStepTable[step_idx].a.limbs);
            }
            
            jac_batch[i] = w->R;
        }

        batchJacobianToAffineDevice(aff_batch, jac_batch, local_count, scratch_prefix, scratch_inv);

        for (int i = 0; i < local_count; i++) {
            if (DP(aff_batch[i].x, DP_BITS)) {
                uint32_t idx = atomicAdd(d_dp_count, 1);
                // Hard limit the buffer to 1 million (or as configured in host) to avoid overflow
                if (idx < exp_steps) {
                    DeviceWalkState* w = &d_walkers[start_idx + i];
                    for(int k=0; k<4; k++) d_dp_buffer[idx].x[k] = aff_batch[i].x[k];
                    d_dp_buffer[idx].a = w->a;
                    d_dp_buffer[idx].b = w->b;
                    d_dp_buffer[idx].walk_id = w->walk_id;
                } else {
                    // Revert the counter if we hit the limit, wait for host to flush
                    atomicSub(d_dp_count, 1); 
                }
            }
        }
    }
}

void launch_lambda_kernel(
    DeviceWalkState* d_walkers,
    DPResult* d_dp_buffer,
    uint32_t* d_dp_count,
    const StepLocal* d_localStepTable,
    uint32_t N_STEPS,
    int DP_BITS,
    int total_walkers,
    unsigned long long* iters,
    int key_range
) {
    const int walkers_per_thread = 8;
    int num_threads = (total_walkers + walkers_per_thread - 1) / walkers_per_thread;
    
    int block_size = 256;
    int grid_size = (num_threads + block_size - 1) / block_size;

#ifdef __HIPCC__
    lambda_walk_kernel<<<grid_size, block_size>>>(
        d_walkers,
        d_dp_buffer,
        d_dp_count,
        d_localStepTable,
        N_STEPS,
        DP_BITS,
        walkers_per_thread,
        total_walkers,
        key_range
    );
#else
    for (int tid = 0; tid < num_threads; tid++) {
        blockIdx.x = tid / block_size;
        threadIdx.x = tid % block_size;
        blockDim.x = block_size;
        
        lambda_walk_kernel(
            d_walkers,
            d_dp_buffer,
            d_dp_count,
            d_localStepTable,
            N_STEPS,
            DP_BITS,
            walkers_per_thread,
            total_walkers,
            key_range
        );
    }
#endif
    
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "HIP kernel failed: " << hipGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // We update the iteration count on the host. 256 steps per kernel launch.
    *iters += (unsigned long long)total_walkers * 256;
}
