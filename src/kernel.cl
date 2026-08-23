#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

typedef struct { uint v[9]; } fe29_t;
typedef struct { fe29_t X; fe29_t Y; fe29_t Z; int infinity; } ECPointJacobian_cl;
typedef struct { ECPointJacobian_cl point; ulong a[4]; } Step_cl;
typedef struct { ulong a[4]; ulong b[4]; ECPointJacobian_cl R; uint walk_id; uint snapshot_steps; ulong snapshot_x[4]; uint state; } WalkState_cl;
typedef struct { ulong a[4]; ulong b[4]; ulong x[4]; uint walk_id; } DPEntry_cl;

#if defined(__OPENCL_VERSION__) || defined(__OPENCL_C_VERSION__)

__constant uint P_CONST[9] = { 0x1FFFFC2F, 0x1FFFFFF7, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x00FFFFFF };

inline fe29_t fe29_add(fe29_t a, fe29_t b) {
    fe29_t r;
    for(int i=0; i<9; i++) r.v[i] = a.v[i] + b.v[i];
    return r;
}

inline fe29_t fe29_sub(fe29_t a, fe29_t b) {
    fe29_t r;
    for(int i=0; i<8; i++) r.v[i] = a.v[i] + 0x3FFFFFFE - b.v[i];
    r.v[8] = a.v[8] + 0x1FFFFFF - b.v[8];
    return r;
}

inline fe29_t fe29_mul_int(fe29_t a, uint b) {
    fe29_t r;
    for(int i=0; i<9; i++) r.v[i] = a.v[i] * b;
    return r;
}

inline fe29_t fe29_neg(fe29_t a) {
    fe29_t p_val;
    for(int i=0; i<9; i++) p_val.v[i] = P_CONST[i];
    return fe29_sub(p_val, a);
}

inline fe29_t fe29_reduce(ulong *c) {
    uint r[9];
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
    fe29_t res;
    for(int i=0; i<9; i++) res.v[i] = (uint)c[i];
    return res;
}
inline fe29_t fe29_mul(fe29_t a, fe29_t b) {    ulong a0 = a.v[0];    ulong b0 = b.v[0];    ulong a1 = a.v[1];    ulong b1 = b.v[1];    ulong a2 = a.v[2];    ulong b2 = b.v[2];    ulong a3 = a.v[3];    ulong b3 = b.v[3];    ulong a4 = a.v[4];    ulong b4 = b.v[4];    ulong a5 = a.v[5];    ulong b5 = b.v[5];    ulong a6 = a.v[6];    ulong b6 = b.v[6];    ulong a7 = a.v[7];    ulong b7 = b.v[7];    ulong a8 = a.v[8];    ulong b8 = b.v[8];    ulong c[9] = {0};    c[0] += a0 * b0;    c[1] += a0 * b1;    c[2] += a0 * b2;    c[3] += a0 * b3;    c[4] += a0 * b4;    c[5] += a0 * b5;    c[6] += a0 * b6;    c[7] += a0 * b7;    c[8] += a0 * b8;    c[1] += a1 * b0;    c[2] += a1 * b1;    c[3] += a1 * b2;    c[4] += a1 * b3;    c[5] += a1 * b4;    c[6] += a1 * b5;    c[7] += a1 * b6;    c[8] += a1 * b7;    c[0] += (a1 * b8) * 0x1000003D1;    c[2] += a2 * b0;    c[3] += a2 * b1;    c[4] += a2 * b2;    c[5] += a2 * b3;    c[6] += a2 * b4;    c[7] += a2 * b5;    c[8] += a2 * b6;    c[0] += (a2 * b7) * 0x1000003D1;    c[1] += (a2 * b8) * 0x1000003D1;    c[3] += a3 * b0;    c[4] += a3 * b1;    c[5] += a3 * b2;    c[6] += a3 * b3;    c[7] += a3 * b4;    c[8] += a3 * b5;    c[0] += (a3 * b6) * 0x1000003D1;    c[1] += (a3 * b7) * 0x1000003D1;    c[2] += (a3 * b8) * 0x1000003D1;    c[4] += a4 * b0;    c[5] += a4 * b1;    c[6] += a4 * b2;    c[7] += a4 * b3;    c[8] += a4 * b4;    c[0] += (a4 * b5) * 0x1000003D1;    c[1] += (a4 * b6) * 0x1000003D1;    c[2] += (a4 * b7) * 0x1000003D1;    c[3] += (a4 * b8) * 0x1000003D1;    c[5] += a5 * b0;    c[6] += a5 * b1;    c[7] += a5 * b2;    c[8] += a5 * b3;    c[0] += (a5 * b4) * 0x1000003D1;    c[1] += (a5 * b5) * 0x1000003D1;    c[2] += (a5 * b6) * 0x1000003D1;    c[3] += (a5 * b7) * 0x1000003D1;    c[4] += (a5 * b8) * 0x1000003D1;    c[6] += a6 * b0;    c[7] += a6 * b1;    c[8] += a6 * b2;    c[0] += (a6 * b3) * 0x1000003D1;    c[1] += (a6 * b4) * 0x1000003D1;    c[2] += (a6 * b5) * 0x1000003D1;    c[3] += (a6 * b6) * 0x1000003D1;    c[4] += (a6 * b7) * 0x1000003D1;    c[5] += (a6 * b8) * 0x1000003D1;    c[7] += a7 * b0;    c[8] += a7 * b1;    c[0] += (a7 * b2) * 0x1000003D1;    c[1] += (a7 * b3) * 0x1000003D1;    c[2] += (a7 * b4) * 0x1000003D1;    c[3] += (a7 * b5) * 0x1000003D1;    c[4] += (a7 * b6) * 0x1000003D1;    c[5] += (a7 * b7) * 0x1000003D1;    c[6] += (a7 * b8) * 0x1000003D1;    c[8] += a8 * b0;    c[0] += (a8 * b1) * 0x1000003D1;    c[1] += (a8 * b2) * 0x1000003D1;    c[2] += (a8 * b3) * 0x1000003D1;    c[3] += (a8 * b4) * 0x1000003D1;    c[4] += (a8 * b5) * 0x1000003D1;    c[5] += (a8 * b6) * 0x1000003D1;    c[6] += (a8 * b7) * 0x1000003D1;    c[7] += (a8 * b8) * 0x1000003D1;    return fe29_reduce(c);}
inline fe29_t fe29_sqr(fe29_t a) {    return fe29_mul(a, a);}
inline fe29_t fe29_inv(fe29_t a) {
    fe29_t x2 = fe29_sqr(a);
    x2 = fe29_mul(x2, a);
    fe29_t x3 = fe29_sqr(x2);
    x3 = fe29_mul(x3, a);
    fe29_t x4 = fe29_sqr(x3);
    x4 = fe29_mul(x4, a);
    fe29_t x8 = x4;
    for(int i=0; i<4; i++) x8 = fe29_sqr(x8);
    x8 = fe29_mul(x8, x4);
    fe29_t x16 = x8;
    for(int i=0; i<8; i++) x16 = fe29_sqr(x16);
    x16 = fe29_mul(x16, x8);
    fe29_t x32 = x16;
    for(int i=0; i<16; i++) x32 = fe29_sqr(x32);
    x32 = fe29_mul(x32, x16);
    fe29_t x64 = x32;
    for(int i=0; i<32; i++) x64 = fe29_sqr(x64);
    x64 = fe29_mul(x64, x32);
    fe29_t x64_2 = x64;
    for(int i=0; i<64; i++) x64_2 = fe29_sqr(x64_2);
    fe29_t x128 = fe29_mul(x64_2, x64);
    fe29_t x256 = x128;
    for(int i=0; i<128; i++) x256 = fe29_sqr(x256);
    x256 = fe29_mul(x256, x128);
    
    fe29_t x = x64;
    for(int i=0; i<64; i++) x = fe29_sqr(x);
    x = fe29_mul(x, x64);
    for(int i=0; i<128; i++) x = fe29_sqr(x);
    x = fe29_mul(x, x128);
    for(int i=0; i<32; i++) x = fe29_sqr(x);
    
    fe29_t t = x; 
    uint rem = 0xFFFFFC2D;
    
    fe29_t r_fin;
    for(int i=0; i<9; i++) r_fin.v[i] = 0;
    r_fin.v[0] = 1; 
    
    fe29_t base = a;
    for(int i=0; i<32; i++) {
        if((rem >> i) & 1) {
            r_fin = fe29_mul(r_fin, base);
        }
        base = fe29_sqr(base);
    }
    
    return fe29_mul(t, r_fin);
}
inline void jacobianAdd(ECPointJacobian_cl *R, ECPointJacobian_cl P, Step_cl Q) {
    if (Q.point.infinity) { *R = P; return; }
    if (P.infinity) { *R = Q.point; return; }

    fe29_t z2 = fe29_sqr(P.Z);
    fe29_t z3 = fe29_mul(P.Z, z2);
    fe29_t u1 = P.X;
    fe29_t u2 = fe29_mul(Q.point.X, z2);
    fe29_t s1 = P.Y;
    fe29_t s2 = fe29_mul(Q.point.Y, z3);
    
    fe29_t h = fe29_sub(u2, u1);
    fe29_t i = fe29_sqr(fe29_mul_int(h, 2));
    fe29_t j = fe29_mul(h, i);
    fe29_t r = fe29_mul_int(fe29_sub(s2, s1), 2);
    fe29_t v = fe29_mul(u1, i);

    R->X = fe29_sub(fe29_sqr(r), j);
    R->X = fe29_sub(R->X, fe29_mul_int(v, 2));

    R->Y = fe29_sub(v, R->X);
    R->Y = fe29_mul(R->Y, r);
    R->Y = fe29_sub(R->Y, fe29_mul_int(fe29_mul(s1, j), 2));

    R->Z = fe29_mul(P.Z, h);
    R->Z = fe29_mul_int(R->Z, 2);
    R->infinity = 0;
}
inline ulong murmur_hash3(ulong x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdul;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ul;
    x ^= x >> 33;
    return x;
}
inline void scalarAdd(ulong* a, const ulong* b) {
    ulong carry = 0;
    for(int i=0; i<4; i++) {
        ulong sum = a[i] + b[i] + carry;
        if(carry) carry = (sum <= a[i]);
        else carry = (sum < a[i]);
        a[i] = sum;
    }
}
inline void scalarSub(ulong* a, const ulong* b) {
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
    uint dp_max_count
) {
    int id = get_global_id(0);
    WalkState_cl w = walkers[id];

    for(uint step = 0; step < n_steps_to_run; step++) {
        // 1. Invert Z to get Affine X and Y
        fe29_t Zinv = fe29_inv(w.R.Z);
        fe29_t Zinv2 = fe29_sqr(Zinv);
        fe29_t Zinv3 = fe29_mul(Zinv2, Zinv);
        fe29_t Xaff = fe29_mul(w.R.X, Zinv2);
        fe29_t Yaff = fe29_mul(w.R.Y, Zinv3);
        
        // 2. Reduce Xaff to 64-bit limbs for hashing and DP
        ulong c[9];
        for(int i=0; i<9; i++) c[i] = Xaff.v[i];
        fe29_t Xaff_reduced = fe29_reduce(c);
        
        ulong X64[4] = {0,0,0,0};
        X64[0] = Xaff_reduced.v[0] | ((ulong)Xaff_reduced.v[1] << 29) | (((ulong)Xaff_reduced.v[2] & 0x3F) << 58);
        X64[1] = ((ulong)Xaff_reduced.v[2] >> 6) | ((ulong)Xaff_reduced.v[3] << 23) | (((ulong)Xaff_reduced.v[4] & 0x1FFFF) << 52);
        X64[2] = ((ulong)Xaff_reduced.v[4] >> 17) | ((ulong)Xaff_reduced.v[5] << 12) | (((ulong)Xaff_reduced.v[6] & 0x7FFFFFF) << 41);
        X64[3] = ((ulong)Xaff_reduced.v[6] >> 27) | ((ulong)Xaff_reduced.v[7] << 2) | ((ulong)Xaff_reduced.v[8] << 31);
        
        // 3. Negation Map parity
        for(int i=0; i<9; i++) c[i] = Yaff.v[i];
        fe29_t Yaff_reduced = fe29_reduce(c);
        bool negate = Yaff_reduced.v[0] & 1;
        
        // 4. Calculate step index
        ulong combined = X64[0] ^ (X64[1] << 1) ^ (X64[2] << 2) ^ (X64[3] << 3);
        uint step_idx = murmur_hash3(combined) % step_table_size;
        
        // 5. Add step point
        Step_cl step_point = stepTable[step_idx];
        if (negate) {
            step_point.point.Y = fe29_neg(step_point.point.Y);
            scalarSub(w.a, step_point.a);
        } else {
            scalarAdd(w.a, step_point.a);
        }
        
        jacobianAdd(&w.R, w.R, step_point);
        w.snapshot_steps++;
        
        // 6. DP Check
        ulong dp_mask = (1ul << dp_bits) - 1;
        if ((X64[0] & dp_mask) == 0) {
            uint idx = atomic_inc(dp_count);
            if (idx < dp_max_count) {
                DPEntry_cl dp;
                for(int k=0; k<4; k++) dp.a[k] = w.a[k];
                for(int k=0; k<4; k++) dp.b[k] = w.b[k];
                for(int k=0; k<4; k++) dp.x[k] = X64[k];
                dp.walk_id = id;
                dp_buffer[idx] = dp;
            }
            w.snapshot_steps = 0;
            w.state = 1; // DP found
            walkers[id] = w;
            return;
        }
    }
    
    walkers[id] = w;
}
#endif