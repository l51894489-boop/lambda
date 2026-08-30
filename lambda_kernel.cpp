#include "lambda_kernel.h"
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__) || defined(USE_NVCC)
    #include "hip_utils.h"
#else
    #ifndef __shared__
        #define __shared__ 
    #endif
    
    #ifndef __syncthreads
        inline void __syncthreads() {}
    #endif

    #ifndef __device__
        #define __device__
    #endif

    #ifndef __global__
        #define __global__
    #endif
#endif


// Pseudo-Random Branch Branch
__device__ inline uint32_t get_step_idx(const uint64_t* aff_x, uint32_t N_STEPS) {
    return (aff_x[0] ^ aff_x[1] ^ aff_x[2] ^ aff_x[3]) & (N_STEPS - 1);
}

// Distinguished Point Check
__device__ inline bool DP(const uint64_t* aff_x, int DP_BITS) {
    if (DP_BITS <= 0) return false;
    uint32_t mask = (1U << DP_BITS) - 1;
    return (aff_x[0] & mask) == 0;
}

__global__ void lambda_walk_kernel(
    uint64_t* d_walkers_X,
    uint64_t* d_walkers_Y,
    uint256_t* d_walkers_a,
    uint256_t* d_walkers_b,
    uint32_t* d_walkers_id,
    DPResult* d_dp_buffer,
    uint32_t* d_dp_count,
    const StepLocal* d_localStepTable,
    uint32_t N_STEPS,
    int DP_BITS,
    int total_walkers,
    int key_range
) {
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;
    
    bool active = (global_id < total_walkers);
    uint64_t w_X[4] = {0,0,0,0}, w_Y[4] = {0,0,0,0};
    uint64_t w_a[4] = {0,0,0,0}, w_b[4] = {0,0,0,0};
    uint32_t w_walk_id = 0;

    if (active) {
        for(int k=0; k<4; k++) {
            w_X[k] = d_walkers_X[global_id * 4 + k];
            w_Y[k] = d_walkers_Y[global_id * 4 + k];
        }
        for(int k=0; k<4; k++) w_a[k] = d_walkers_a[global_id].limbs[k];
        for(int k=0; k<4; k++) w_b[k] = d_walkers_b[global_id].limbs[k];
        w_walk_id = d_walkers_id[global_id];
    }

    uint64_t exp_steps = 1ULL << (key_range / 2);
    uint32_t max_buffer_size = (exp_steps > 1000000ULL) ? 1000000 : static_cast<uint32_t>(exp_steps);

    const int CHUNK_SIZE = 32;
    int chunk_id = tid / CHUNK_SIZE;
    int lane_id  = tid % CHUNK_SIZE;
    
    __shared__ uint64_t s_Z[256][4];
    __shared__ uint64_t s_prefix[256][4];
    __shared__ uint64_t s_inv[256][4];
    __shared__ StepLocal s_stepTable[256];

    if (tid < N_STEPS) {
        s_stepTable[tid] = d_localStepTable[tid];
    }
    __syncthreads();

    for (int step = 0; step < 256; step++) {
        uint32_t step_idx = 0;
        bool negate = false;
        ECPointJacobian step_point;
        
        if (active) {
            step_idx = get_step_idx(w_X, N_STEPS);
            negate = w_Y[0] & 1;
            step_point = s_stepTable[step_idx].point;
            
            if (negate) {
                uint64_t zero[4] = {0, 0, 0, 0};
                modSubP(step_point.Y, zero, step_point.Y);
            }
            
            modSubP(s_Z[tid], w_X, step_point.X);
            if (scalarIsZero(s_Z[tid])) {
                s_Z[tid][0] = 0x00000001000003D1ULL;
            }
        } else {
            s_Z[tid][0] = 0x00000001000003D1ULL;
            s_Z[tid][1] = 0; s_Z[tid][2] = 0; s_Z[tid][3] = 0;
        }
        __syncthreads(); 

        if (lane_id == 0) {
            int base = chunk_id * CHUNK_SIZE;
            
            for(int k=0; k<4; k++) s_prefix[base][k] = s_Z[base][k];
            for (int i = 1; i < CHUNK_SIZE; i++) {
                modMulMontP(s_prefix[base + i], s_prefix[base + i - 1], s_Z[base + i]);
            }
            
            uint64_t P_MINUS_2[4];
            P_MINUS_2[0] = 0xFFFFFFFEFFFFFC2DULL;
            P_MINUS_2[1] = 0xFFFFFFFFFFFFFFFFULL;
            P_MINUS_2[2] = 0xFFFFFFFFFFFFFFFFULL;
            P_MINUS_2[3] = 0xFFFFFFFFFFFFFFFFULL;

            modExpMontP(s_inv[base + CHUNK_SIZE - 1], s_prefix[base + CHUNK_SIZE - 1], P_MINUS_2);
            
            for (int i = CHUNK_SIZE - 1; i > 0; i--) {
                modMulMontP(s_inv[base + i - 1], s_inv[base + i], s_Z[base + i]);
                modMulMontP(s_inv[base + i], s_inv[base + i], s_prefix[base + i - 1]);
            }
        }
        __syncthreads();

        if (active) {
            if (DP(w_X, DP_BITS)) {
                uint32_t idx = atomicAdd(d_dp_count, 1);
                if (idx < max_buffer_size) {
                    for(int k=0; k<4; k++) d_dp_buffer[idx].x[k] = w_X[k];
                    for(int k=0; k<4; k++) d_dp_buffer[idx].a.limbs[k] = w_a[k];
                    for(int k=0; k<4; k++) d_dp_buffer[idx].b.limbs[k] = w_b[k];
                    d_dp_buffer[idx].walk_id = w_walk_id;
                } else {
                    atomicSub(d_dp_count, 1);
                }
            }

            uint64_t dy[4];
            modSubP(dy, w_Y, step_point.Y);
            
            uint64_t lambda[4];
            modMulMontP(lambda, dy, s_inv[tid]);
            
            uint64_t lambda_sq[4];
            modMulMontP(lambda_sq, lambda, lambda);
            
            uint64_t new_X[4];
            modSubP(new_X, lambda_sq, w_X);
            modSubP(new_X, new_X, step_point.X);
            
            uint64_t new_Y[4];
            modSubP(new_Y, w_X, new_X);
            modMulMontP(new_Y, lambda, new_Y);
            modSubP(new_Y, new_Y, w_Y);
            
            for(int k=0; k<4; k++) w_X[k] = new_X[k];
            for(int k=0; k<4; k++) w_Y[k] = new_Y[k];
            
            if (negate) {
                scalarSub(w_a, w_a, s_stepTable[step_idx].a.limbs);
            } else {
                scalarAdd(w_a, w_a, s_stepTable[step_idx].a.limbs);
            }
        }
    }
    
    if (active) {
        for(int k=0; k<4; k++) {
            d_walkers_X[global_id * 4 + k] = w_X[k];
            d_walkers_Y[global_id * 4 + k] = w_Y[k];
        }
        for(int k=0; k<4; k++) d_walkers_a[global_id].limbs[k] = w_a[k];
        for(int k=0; k<4; k++) d_walkers_b[global_id].limbs[k] = w_b[k];
    }
}

void launch_lambda_kernel(
    uint64_t* d_walkers_X,
    uint64_t* d_walkers_Y,
    uint256_t* d_walkers_a,
    uint256_t* d_walkers_b,
    uint32_t* d_walkers_id,
    DPResult* d_dp_buffer,
    uint32_t* d_dp_count,
    const StepLocal* d_localStepTable,
    uint32_t N_STEPS,
    int DP_BITS,
    int total_walkers,
    unsigned long long* iters,
    int key_range
) {
    const int block_size = 256; 
    const int grid_size = (total_walkers + block_size - 1) / block_size;

#if defined(__HIPCC__) || defined(__CUDACC__) || defined(USE_NVCC)
    lambda_walk_kernel<<<grid_size, block_size>>>(
        d_walkers_X,
        d_walkers_Y,
        d_walkers_a,
        d_walkers_b,
        d_walkers_id,
        d_dp_buffer,
        d_dp_count,
        d_localStepTable,
        N_STEPS,
        DP_BITS,
        total_walkers,
        key_range
    );
#else
    for (int tid = 0; tid < total_walkers; tid++) {}
#endif

#if defined(__HIPCC__) || defined(__CUDACC__) || defined(USE_NVCC)
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("GPU KERNEL LAUNCH ERROR: %s\n", hipGetErrorString(err));
        return; 
    }
    *iters += (unsigned long long)total_walkers * 256;
#else
    printf("\n[ERROR] EXECUTANDO EM MODO 'make mock' (CPU) MAS O FALLBACK DA CPU ESTÁ VAZIO!\n");
    printf("[ERROR] Nenhuma operação está sendo feita. Compile com 'make' nativo na sua máquina de produção.\n");
    exit(1);
#endif
}
