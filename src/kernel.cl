#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#define BATCH_SIZE 32

typedef struct { uint v[9]; } fe29_t;
typedef struct { fe29_t X; fe29_t Y; int infinity; } ECPointAffine_cl;
typedef struct { ECPointAffine_cl point; ulong a[4]; } Step_cl;
typedef struct { ulong a[4]; ulong b[4]; ECPointAffine_cl R; uint walk_id; uint snapshot_steps; ulong snapshot_x[4]; uint state; } WalkState_cl;
typedef struct { ulong a[4]; ulong b[4]; ulong x[4]; uint walk_id; } DPEntry_cl;

#if defined(__OPENCL_VERSION__) || defined(__OPENCL_C_VERSION__)

__constant uint P_CONST[9] = { 0x1FFFFC2F, 0x1FFFFFF7, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x00FFFFFF };

inline void fe29_add(__private fe29_t *r, __private const fe29_t *a, __private const fe29_t *b) {
    for(int i=0; i<9; i++) r->v[i] = a->v[i] + b->v[i];
}

inline void fe29_sub(__private fe29_t *r, __private const fe29_t *a, __private const fe29_t *b) {
    for(int i=0; i<8; i++) r->v[i] = a->v[i] + 0x3FFFFFFE - b->v[i];
    r->v[8] = a->v[8] + 0x1FFFFFF - b->v[8];
}

inline void fe29_mul_int(__private fe29_t *r, __private const fe29_t *a, uint b) {
    for(int i=0; i<9; i++) r->v[i] = a->v[i] * b;
}

inline void fe29_neg(__private fe29_t *r, __private const fe29_t *a) {
    for(int i=0; i<8; i++) r->v[i] = P_CONST[i] + 0x3FFFFFFE - a->v[i];
    r->v[8] = P_CONST[8] + 0x1FFFFFF - a->v[8];
}

inline void fe29_reduce(__private fe29_t *res, __private ulong *c) {
    ulong c8 = c[8];
    ulong top = c8 >> 24;
    c[8] = c8 & 0xFFFFFF;
    c[0] += top * 0x1000003D1;
    for(int i=0; i<8; i++) {
        ulong carry = c[i] >> 29;
        c[i] &= 0x1FFFFFFF;
        c[i+1] += carry;
    }
    top = c[8] >> 24;
    c[8] &= 0xFFFFFF;
    c[0] += top * 0x1000003D1;
    ulong carry = c[0] >> 29;
    c[0] &= 0x1FFFFFFF;
    c[1] += carry;
    for(int i=0; i<9; i++) res->v[i] = (uint)c[i];
}

inline void fe29_mul(__private fe29_t *r, __private const fe29_t *a, __private const fe29_t *b) {
    ulong c[9] = {0,0,0,0,0,0,0,0,0};
    c[0] += (ulong)a->v[0] * (ulong)b->v[0];    c[1] += (ulong)a->v[0] * (ulong)b->v[1];    c[2] += (ulong)a->v[0] * (ulong)b->v[2];    c[3] += (ulong)a->v[0] * (ulong)b->v[3];    c[4] += (ulong)a->v[0] * (ulong)b->v[4];    c[5] += (ulong)a->v[0] * (ulong)b->v[5];    c[6] += (ulong)a->v[0] * (ulong)b->v[6];    c[7] += (ulong)a->v[0] * (ulong)b->v[7];    c[8] += (ulong)a->v[0] * (ulong)b->v[8];    c[1] += (ulong)a->v[1] * (ulong)b->v[0];    c[2] += (ulong)a->v[1] * (ulong)b->v[1];    c[3] += (ulong)a->v[1] * (ulong)b->v[2];    c[4] += (ulong)a->v[1] * (ulong)b->v[3];    c[5] += (ulong)a->v[1] * (ulong)b->v[4];    c[6] += (ulong)a->v[1] * (ulong)b->v[5];    c[7] += (ulong)a->v[1] * (ulong)b->v[6];    c[8] += (ulong)a->v[1] * (ulong)b->v[7];    c[0] += ((ulong)a->v[1] * (ulong)b->v[8]) * 0x1000003D1;    c[2] += (ulong)a->v[2] * (ulong)b->v[0];    c[3] += (ulong)a->v[2] * (ulong)b->v[1];    c[4] += (ulong)a->v[2] * (ulong)b->v[2];    c[5] += (ulong)a->v[2] * (ulong)b->v[3];    c[6] += (ulong)a->v[2] * (ulong)b->v[4];    c[7] += (ulong)a->v[2] * (ulong)b->v[5];    c[8] += (ulong)a->v[2] * (ulong)b->v[6];    c[0] += ((ulong)a->v[2] * (ulong)b->v[7]) * 0x1000003D1;    c[1] += ((ulong)a->v[2] * (ulong)b->v[8]) * 0x1000003D1;    c[3] += (ulong)a->v[3] * (ulong)b->v[0];    c[4] += (ulong)a->v[3] * (ulong)b->v[1];    c[5] += (ulong)a->v[3] * (ulong)b->v[2];    c[6] += (ulong)a->v[3] * (ulong)b->v[3];    c[7] += (ulong)a->v[3] * (ulong)b->v[4];    c[8] += (ulong)a->v[3] * (ulong)b->v[5];    c[0] += ((ulong)a->v[3] * (ulong)b->v[6]) * 0x1000003D1;    c[1] += ((ulong)a->v[3] * (ulong)b->v[7]) * 0x1000003D1;    c[2] += ((ulong)a->v[3] * (ulong)b->v[8]) * 0x1000003D1;    c[4] += (ulong)a->v[4] * (ulong)b->v[0];    c[5] += (ulong)a->v[4] * (ulong)b->v[1];    c[6] += (ulong)a->v[4] * (ulong)b->v[2];    c[7] += (ulong)a->v[4] * (ulong)b->v[3];    c[8] += (ulong)a->v[4] * (ulong)b->v[4];    c[0] += ((ulong)a->v[4] * (ulong)b->v[5]) * 0x1000003D1;    c[1] += ((ulong)a->v[4] * (ulong)b->v[6]) * 0x1000003D1;    c[2] += ((ulong)a->v[4] * (ulong)b->v[7]) * 0x1000003D1;    c[3] += ((ulong)a->v[4] * (ulong)b->v[8]) * 0x1000003D1;    c[5] += (ulong)a->v[5] * (ulong)b->v[0];    c[6] += (ulong)a->v[5] * (ulong)b->v[1];    c[7] += (ulong)a->v[5] * (ulong)b->v[2];    c[8] += (ulong)a->v[5] * (ulong)b->v[3];    c[0] += ((ulong)a->v[5] * (ulong)b->v[4]) * 0x1000003D1;    c[1] += ((ulong)a->v[5] * (ulong)b->v[5]) * 0x1000003D1;    c[2] += ((ulong)a->v[5] * (ulong)b->v[6]) * 0x1000003D1;    c[3] += ((ulong)a->v[5] * (ulong)b->v[7]) * 0x1000003D1;    c[4] += ((ulong)a->v[5] * (ulong)b->v[8]) * 0x1000003D1;    c[6] += (ulong)a->v[6] * (ulong)b->v[0];    c[7] += (ulong)a->v[6] * (ulong)b->v[1];    c[8] += (ulong)a->v[6] * (ulong)b->v[2];    c[0] += ((ulong)a->v[6] * (ulong)b->v[3]) * 0x1000003D1;    c[1] += ((ulong)a->v[6] * (ulong)b->v[4]) * 0x1000003D1;    c[2] += ((ulong)a->v[6] * (ulong)b->v[5]) * 0x1000003D1;    c[3] += ((ulong)a->v[6] * (ulong)b->v[6]) * 0x1000003D1;    c[4] += ((ulong)a->v[6] * (ulong)b->v[7]) * 0x1000003D1;    c[5] += ((ulong)a->v[6] * (ulong)b->v[8]) * 0x1000003D1;    c[7] += (ulong)a->v[7] * (ulong)b->v[0];    c[8] += (ulong)a->v[7] * (ulong)b->v[1];    c[0] += ((ulong)a->v[7] * (ulong)b->v[2]) * 0x1000003D1;    c[1] += ((ulong)a->v[7] * (ulong)b->v[3]) * 0x1000003D1;    c[2] += ((ulong)a->v[7] * (ulong)b->v[4]) * 0x1000003D1;    c[3] += ((ulong)a->v[7] * (ulong)b->v[5]) * 0x1000003D1;    c[4] += ((ulong)a->v[7] * (ulong)b->v[6]) * 0x1000003D1;    c[5] += ((ulong)a->v[7] * (ulong)b->v[7]) * 0x1000003D1;    c[6] += ((ulong)a->v[7] * (ulong)b->v[8]) * 0x1000003D1;    c[8] += (ulong)a->v[8] * (ulong)b->v[0];    c[0] += ((ulong)a->v[8] * (ulong)b->v[1]) * 0x1000003D1;    c[1] += ((ulong)a->v[8] * (ulong)b->v[2]) * 0x1000003D1;    c[2] += ((ulong)a->v[8] * (ulong)b->v[3]) * 0x1000003D1;    c[3] += ((ulong)a->v[8] * (ulong)b->v[4]) * 0x1000003D1;    c[4] += ((ulong)a->v[8] * (ulong)b->v[5]) * 0x1000003D1;    c[5] += ((ulong)a->v[8] * (ulong)b->v[6]) * 0x1000003D1;    c[6] += ((ulong)a->v[8] * (ulong)b->v[7]) * 0x1000003D1;    c[7] += ((ulong)a->v[8] * (ulong)b->v[8]) * 0x1000003D1;
    fe29_reduce(r, c);
}

inline void fe29_sqr(__private fe29_t *r, __private const fe29_t *a) {
    fe29_mul(r, a, a);
}

inline void fe29_inv(__private fe29_t *r, __private const fe29_t *a) {
    fe29_t x2; fe29_sqr(&x2, a); fe29_mul(&x2, &x2, a);
    fe29_t x3; fe29_sqr(&x3, &x2); fe29_mul(&x3, &x3, a);
    fe29_t x4; fe29_sqr(&x4, &x3); fe29_mul(&x4, &x4, a);
    fe29_t x8 = x4;
    for(int i=0; i<4; i++) fe29_sqr(&x8, &x8);
    fe29_mul(&x8, &x8, &x4);
    fe29_t x16 = x8;
    for(int i=0; i<8; i++) fe29_sqr(&x16, &x16);
    fe29_mul(&x16, &x16, &x8);
    fe29_t x32 = x16;
    for(int i=0; i<16; i++) fe29_sqr(&x32, &x32);
    fe29_mul(&x32, &x32, &x16);
    fe29_t x64 = x32;
    for(int i=0; i<32; i++) fe29_sqr(&x64, &x64);
    fe29_mul(&x64, &x64, &x32);
    fe29_t x64_2 = x64;
    for(int i=0; i<64; i++) fe29_sqr(&x64_2, &x64_2);
    fe29_t x128; fe29_mul(&x128, &x64_2, &x64);
    fe29_t x256 = x128;
    for(int i=0; i<128; i++) fe29_sqr(&x256, &x256);
    fe29_mul(&x256, &x256, &x128);
    
    fe29_t x = x64;
    for(int i=0; i<64; i++) fe29_sqr(&x, &x);
    fe29_mul(&x, &x, &x64);
    for(int i=0; i<128; i++) fe29_sqr(&x, &x);
    fe29_mul(&x, &x, &x128);
    for(int i=0; i<32; i++) fe29_sqr(&x, &x);
    
    fe29_t t = x;
    uint rem = 0xFFFFFC2D;
    
    fe29_t r_fin;
    for(int i=0; i<9; i++) r_fin.v[i] = 0;
    r_fin.v[0] = 1; 
    
    fe29_t base;
    for(int i=0; i<9; i++) base.v[i] = a->v[i];
    
    for(int i=0; i<32; i++) {
        if((rem >> i) & 1) {
            fe29_mul(&r_fin, &r_fin, &base);
        }
        fe29_sqr(&base, &base);
    }
    
    fe29_mul(r, &t, &r_fin);
}

inline void fe29_set_one(__private fe29_t *r) {
    for(int i=0; i<9; i++) r->v[i] = 0;
    r->v[0] = 1;
}

inline ulong murmur_hash3(ulong x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdul;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ul;
    x ^= x >> 33;
    return x;
}

inline void scalarAdd(__private ulong* a, __private const ulong* b) {
    ulong carry = 0;
    for(int i=0; i<4; i++) {
        ulong sum = a[i] + b[i] + carry;
        if(carry) carry = (sum <= a[i]);
        else carry = (sum < a[i]);
        a[i] = sum;
    }
}
inline void scalarSub(__private ulong* a, __private const ulong* b) {
    ulong borrow = 0;
    for(int i=0; i<4; i++) {
        ulong ai = a[i];
        ulong bi = b[i];
        ulong diff = ai - bi - borrow;
        if(borrow) borrow = (ai <= bi);
        else borrow = (ai < bi);
        a[i] = diff;
    }
}

__kernel void kangaroo_walk(
    __global WalkState_cl *walkers,
    __global Step_cl *stepTable,
    uint n_steps_to_run,
    uint dp_bits,
    __global DPEntry_cl *dp_buffer,
    __global uint *dp_count,
    uint step_table_size,
    uint dp_max_count,
    __global fe29_t *batchInvBuffer
) {
    int id = get_global_id(0);
    int batch_start = id * BATCH_SIZE;
    
    // We do n_steps_to_run times BATCH_SIZE steps
    for(uint step = 0; step < n_steps_to_run; step++) {
        
        fe29_t cumulative_prod;
        fe29_set_one(&cumulative_prod);
        
        uint local_step_idx[BATCH_SIZE];
        bool local_negate[BATCH_SIZE];
        
        // 1. FORWARD PASS: accumulate differences
        for (int i = 0; i < BATCH_SIZE; i++) {
            int w_id = batch_start + i;
            
            // Wait, if a walker is at DP, we should skip it or handle it. 
            // In OpenCL, if a walker found DP, state == 1, we can just skip it here.
            uint state = walkers[w_id].state;
            if (state == 1) {
                // To keep the batch unbroken, we multiply by 1
                fe29_t one; fe29_set_one(&one);
                batchInvBuffer[w_id] = cumulative_prod; 
                fe29_mul(&cumulative_prod, &cumulative_prod, &one);
                continue;
            }
            
            fe29_t X;
            for(int k=0; k<9; k++) X.v[k] = walkers[w_id].R.X.v[k];
            fe29_t Y;
            for(int k=0; k<9; k++) Y.v[k] = walkers[w_id].R.Y.v[k];
            
            // Reduce X to 64 bits to find jump point
            ulong c[9];
            for(int k=0; k<9; k++) c[k] = X.v[k];
            fe29_t X_reduced; fe29_reduce(&X_reduced, c);
            
            ulong X64 = X_reduced.v[0] | ((ulong)X_reduced.v[1] << 29) | (((ulong)X_reduced.v[2] & 0x3F) << 58);
            
            // DP check on X64
            ulong dp_mask = (1ul << dp_bits) - 1;
            if ((X64 & dp_mask) == 0) {
                // Hit DP! 
                uint dp_idx = atomic_inc(dp_count);
                if (dp_idx < dp_max_count) {
                    for(int k=0; k<4; k++) dp_buffer[dp_idx].a[k] = walkers[w_id].a[k];
                    for(int k=0; k<4; k++) dp_buffer[dp_idx].b[k] = walkers[w_id].b[k];
                    
                    ulong X64_full[4];
                    X64_full[0] = X64;
                    X64_full[1] = ((ulong)X_reduced.v[2] >> 6) | ((ulong)X_reduced.v[3] << 23) | (((ulong)X_reduced.v[4] & 0x1FFFF) << 52);
                    X64_full[2] = ((ulong)X_reduced.v[4] >> 17) | ((ulong)X_reduced.v[5] << 12) | (((ulong)X_reduced.v[6] & 0x7FFFFFF) << 41);
                    X64_full[3] = ((ulong)X_reduced.v[6] >> 27) | ((ulong)X_reduced.v[7] << 2) | ((ulong)X_reduced.v[8] << 31);
                    
                    for(int k=0; k<4; k++) dp_buffer[dp_idx].x[k] = X64_full[k];
                    dp_buffer[dp_idx].walk_id = w_id;
                }
                walkers[w_id].snapshot_steps = 0;
                walkers[w_id].state = 1; // Mark as done
                
                // Still need to maintain batch product chain
                fe29_t one; fe29_set_one(&one);
                batchInvBuffer[w_id] = cumulative_prod; 
                fe29_mul(&cumulative_prod, &cumulative_prod, &one);
                continue;
            }
            
            // Not a DP. Find jump index.
            ulong X64_full[4];
            X64_full[0] = X64;
            X64_full[1] = ((ulong)X_reduced.v[2] >> 6) | ((ulong)X_reduced.v[3] << 23) | (((ulong)X_reduced.v[4] & 0x1FFFF) << 52);
            X64_full[2] = ((ulong)X_reduced.v[4] >> 17) | ((ulong)X_reduced.v[5] << 12) | (((ulong)X_reduced.v[6] & 0x7FFFFFF) << 41);
            X64_full[3] = ((ulong)X_reduced.v[6] >> 27) | ((ulong)X_reduced.v[7] << 2) | ((ulong)X_reduced.v[8] << 31);
            
            ulong combined = X64_full[0] ^ (X64_full[1] << 1) ^ (X64_full[2] << 2) ^ (X64_full[3] << 3);
            uint step_idx = murmur_hash3(combined) % step_table_size;
            local_step_idx[i] = step_idx;
            
            // Negation map parity
            for(int k=0; k<9; k++) c[k] = Y.v[k];
            fe29_t Y_reduced; fe29_reduce(&Y_reduced, c);
            bool negate = Y_reduced.v[0] & 1;
            local_negate[i] = negate;
            
            // dx = X_0 - X_jmp
            fe29_t Qx;
            for(int k=0; k<9; k++) Qx.v[k] = stepTable[step_idx].point.X.v[k];
            
            fe29_t dx; fe29_sub(&dx, &X, &Qx);
            
            batchInvBuffer[w_id] = cumulative_prod; // Save product up to i-1
            fe29_mul(&cumulative_prod, &cumulative_prod, &dx); // Multiply by dx_i
        }
        
        // 2. BATCH INVERSION
        fe29_inv(&cumulative_prod, &cumulative_prod);
        
        // 3. BACKWARD PASS: compute new points
        for (int i = BATCH_SIZE - 1; i >= 0; i--) {
            int w_id = batch_start + i;
            if (walkers[w_id].state == 1) continue; // Skip resolved DPs
            
            // Inverse of dx_i is: cumulative_prod * batchInvBuffer[w_id]
            fe29_t prev_prod;
            for(int k=0; k<9; k++) prev_prod.v[k] = batchInvBuffer[w_id].v[k];
            
            fe29_t dx_inv; fe29_mul(&dx_inv, &cumulative_prod, &prev_prod);
            
            // We must now update cumulative_prod = cumulative_prod * dx_i
            // So we re-fetch X and Qx
            fe29_t X;
            for(int k=0; k<9; k++) X.v[k] = walkers[w_id].R.X.v[k];
            uint step_idx = local_step_idx[i];
            
            fe29_t Qx;
            for(int k=0; k<9; k++) Qx.v[k] = stepTable[step_idx].point.X.v[k];
            
            fe29_t dx; fe29_sub(&dx, &X, &Qx);
            fe29_mul(&cumulative_prod, &cumulative_prod, &dx); // Restored for i-1
            
            // Affine Point Addition
            fe29_t Y;
            for(int k=0; k<9; k++) Y.v[k] = walkers[w_id].R.Y.v[k];
            
            fe29_t Qy;
            for(int k=0; k<9; k++) Qy.v[k] = stepTable[step_idx].point.Y.v[k];
            
            if (local_negate[i]) {
                fe29_neg(&Qy, &Qy);
            }
            
            // s = (Y_0 - Qy) * dx_inv
            fe29_t s; fe29_sub(&s, &Y, &Qy);
            fe29_mul(&s, &s, &dx_inv);
            
            // X_new = s^2 - X_0 - Qx
            fe29_t X_new; fe29_sqr(&X_new, &s);
            fe29_sub(&X_new, &X_new, &X);
            fe29_sub(&X_new, &X_new, &Qx);
            
            // Y_new = s(X_0 - X_new) - Y_0
            fe29_t Y_new; fe29_sub(&Y_new, &X, &X_new);
            fe29_mul(&Y_new, &Y_new, &s);
            fe29_sub(&Y_new, &Y_new, &Y);
            
            // Save state
            for(int k=0; k<9; k++) walkers[w_id].R.X.v[k] = X_new.v[k];
            for(int k=0; k<9; k++) walkers[w_id].R.Y.v[k] = Y_new.v[k];
            
            // Update scalar a
            ulong Q_a[4];
            for(int k=0; k<4; k++) Q_a[k] = stepTable[step_idx].a[k];
            
            ulong w_a[4];
            for(int k=0; k<4; k++) w_a[k] = walkers[w_id].a[k];
            
            if (local_negate[i]) {
                scalarSub(w_a, Q_a);
            } else {
                scalarAdd(w_a, Q_a);
            }
            for(int k=0; k<4; k++) walkers[w_id].a[k] = w_a[k];
            walkers[w_id].snapshot_steps++;
        }
    }
}
#endif
