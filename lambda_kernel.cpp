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
    uint64_t* d_walkers_s,
    uint256_t* d_walkers_a,
    uint256_t* d_walkers_b,
    uint32_t* d_walkers_id,
    DPResult* d_dp_buffer,
    uint32_t* d_dp_count,
    const StepLocal* d_localStepTable,
    uint32_t N_STEPS,
    int DP_BITS,
    int total_threads,
    int group_size,
    int key_range
) {
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;
    
    if (global_id >= total_threads) return;

    uint64_t exp_steps = 1ULL << (key_range / 2);
    uint32_t max_buffer_size = (exp_steps > 1000000ULL) ? 1000000 : static_cast<uint32_t>(exp_steps);

    // Number of steps per kernel launch. 10 loops of 24 points = 240 steps.
    const int LOOPS = 10;
    
    for (int loop = 0; loop < LOOPS; loop++) {
        uint64_t inverse[4] = {0, 0, 0, 0};
        uint32_t step_indices[32]; // Max group size 32
        bool negates[32];
        
        // ETAPA 1: Computação do DX e acúmulo dos prefixos em Global Memory (L2 Cache)
        for (int group = 0; group < group_size; group++) {
            uint64_t x[4];
            uint64_t y[4];
            
            int idx = group * total_threads + global_id;
            int base_idx = idx * 4;
            
            for (int k = 0; k < 4; k++) {
                x[k] = d_walkers_X[base_idx + k];
                y[k] = d_walkers_Y[base_idx + k];
            }
            
            step_indices[group] = get_step_idx(x, N_STEPS);
            negates[group] = y[0] & 1;
            
            ECPointJacobian step_point = d_localStepTable[step_indices[group]].point;
            
            uint64_t dx[4];
            modSubP(dx, x, step_point.X);
            if (scalarIsZero(dx)) {
                dx[0] = 0x00000001000003D1ULL;
                dx[1] = 0; dx[2] = 0; dx[3] = 0;
            }
            
            if (group == 0) {
                for (int k = 0; k < 4; k++) inverse[k] = dx[k];
            } else {
                modMulMontP(inverse, inverse, dx);
            }
            
            for (int k = 0; k < 4; k++) {
                d_walkers_s[base_idx + k] = inverse[k];
            }
        }
        
        // ETAPA 2: Batch Inversion INDIVIDUAL (Cada thread inverte os seus 24 pontos)
        // Uso de ALU a 100% (Sem warp divergence)
        uint64_t P_MINUS_2[4] = {0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
        modExpMontP(inverse, inverse, P_MINUS_2);
        
        // ETAPA 3: Recuperação da inversão e Acúmulo Afim
        for (int group = group_size - 1; group >= 0; group--) {
            uint64_t x[4];
            uint64_t y[4];
            uint64_t my_dx_inv[4];
            
            int idx = group * total_threads + global_id;
            int base_idx = idx * 4;
            
            for (int k = 0; k < 4; k++) {
                x[k] = d_walkers_X[base_idx + k];
                y[k] = d_walkers_Y[base_idx + k];
            }
            
            ECPointJacobian step_point = d_localStepTable[step_indices[group]].point;
            bool negate = negates[group];
            if (negate) {
                uint64_t zero[4] = {0, 0, 0, 0};
                modSubP(step_point.Y, zero, step_point.Y);
            }
            
            if (group > 0) {
                uint64_t dx[4];
                modSubP(dx, x, step_point.X);
                if (scalarIsZero(dx)) {
                    dx[0] = 0x00000001000003D1ULL;
                    dx[1] = 0; dx[2] = 0; dx[3] = 0;
                }
                
                uint64_t prev_prefix[4];
                int prev_base = ((group - 1) * total_threads + global_id) * 4;
                for (int k = 0; k < 4; k++) prev_prefix[k] = d_walkers_s[prev_base + k];
                
                modMulMontP(my_dx_inv, prev_prefix, inverse);
                modMulMontP(inverse, inverse, dx);
            } else {
                for (int k = 0; k < 4; k++) my_dx_inv[k] = inverse[k];
            }
            
            // DP Check
            if (DP(x, DP_BITS)) {
                uint32_t dp_idx = atomicAdd(d_dp_count, 1);
                if (dp_idx < max_buffer_size) {
                    for(int k=0; k<4; k++) d_dp_buffer[dp_idx].x[k] = x[k];
                    for(int k=0; k<4; k++) d_dp_buffer[dp_idx].a.limbs[k] = d_walkers_a[idx].limbs[k];
                    for(int k=0; k<4; k++) d_dp_buffer[dp_idx].b.limbs[k] = d_walkers_b[idx].limbs[k];
                    d_dp_buffer[dp_idx].walk_id = d_walkers_id[idx];
                } else {
                    atomicSub(d_dp_count, 1);
                }
            }
            
            // Affine Addition
            uint64_t dy[4];
            modSubP(dy, y, step_point.Y);
            
            uint64_t lambda[4];
            modMulMontP(lambda, dy, my_dx_inv);
            
            uint64_t lambda_sq[4];
            modMulMontP(lambda_sq, lambda, lambda);
            
            uint64_t new_X[4];
            modSubP(new_X, lambda_sq, x);
            modSubP(new_X, new_X, step_point.X);
            
            uint64_t new_Y[4];
            modSubP(new_Y, x, new_X);
            modMulMontP(new_Y, lambda, new_Y);
            modSubP(new_Y, new_Y, y);
            
            for (int k = 0; k < 4; k++) {
                d_walkers_X[base_idx + k] = new_X[k];
                d_walkers_Y[base_idx + k] = new_Y[k];
            }
            
            // Atualiza os escalares
            uint256_t w_a;
            for (int k = 0; k < 4; k++) w_a.limbs[k] = d_walkers_a[idx].limbs[k];
            
            if (negate) {
                scalarSub(w_a.limbs, w_a.limbs, d_localStepTable[step_indices[group]].a.limbs);
            } else {
                scalarAdd(w_a.limbs, w_a.limbs, d_localStepTable[step_indices[group]].a.limbs);
            }
            
            for (int k = 0; k < 4; k++) d_walkers_a[idx].limbs[k] = w_a.limbs[k];
        }
    }
}

void launch_lambda_kernel(
    uint64_t* d_walkers_X,
    uint64_t* d_walkers_Y,
    uint64_t* d_walkers_s,
    uint256_t* d_walkers_a,
    uint256_t* d_walkers_b,
    uint32_t* d_walkers_id,
    DPResult* d_dp_buffer,
    uint32_t* d_dp_count,
    const StepLocal* d_localStepTable,
    uint32_t N_STEPS,
    int DP_BITS,
    int total_threads,
    int group_size,
    unsigned long long* iters,
    int key_range
) {
    const int block_size = 256; 
    const int grid_size = (total_threads + block_size - 1) / block_size;

#if defined(__HIPCC__) || defined(__CUDACC__) || defined(USE_NVCC)
    lambda_walk_kernel<<<grid_size, block_size>>>(
        d_walkers_X,
        d_walkers_Y,
        d_walkers_s,
        d_walkers_a,
        d_walkers_b,
        d_walkers_id,
        d_dp_buffer,
        d_dp_count,
        d_localStepTable,
        N_STEPS,
        DP_BITS,
        total_threads,
        group_size,
        key_range
    );
#else
    for (int tid = 0; tid < total_threads; tid++) {}
#endif

#if defined(__HIPCC__) || defined(__CUDACC__) || defined(USE_NVCC)
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("GPU KERNEL LAUNCH ERROR: %s\n", hipGetErrorString(err));
        return; 
    }
    // LOOPS = 10. Total steps = LOOPS * group_size * total_threads
    *iters += (unsigned long long)total_threads * group_size * 10;
#else
    printf("\n[ERROR] EXECUTANDO EM MODO 'make mock' (CPU) MAS O FALLBACK DA CPU ESTÁ VAZIO!\n");
    printf("[ERROR] Nenhuma operação está sendo feita. Compile com 'make' nativo na sua máquina de produção.\n");
    exit(1);
#endif
}
