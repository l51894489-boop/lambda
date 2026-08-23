#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

typedef struct { uint v[9]; } fe29_t;
typedef struct { fe29_t X; fe29_t Y; fe29_t Z; int infinity; } ECPointJacobian_cl;
typedef struct { ECPointJacobian_cl point; ulong a[4]; } Step_cl;
typedef struct { ulong a[4]; ulong b[4]; ECPointJacobian_cl R; uint walk_id; uint snapshot_steps; ulong snapshot_x[4]; uint state; } WalkState_cl;
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

inline void jacobianAdd(__private ECPointJacobian_cl *R, __private const ECPointJacobian_cl *P, __private const Step_cl *Q) {
    if (Q->point.infinity) { 
        for(int i=0; i<9; i++) R->X.v[i] = P->X.v[i];
        for(int i=0; i<9; i++) R->Y.v[i] = P->Y.v[i];
        for(int i=0; i<9; i++) R->Z.v[i] = P->Z.v[i];
        R->infinity = P->infinity; 
        return; 
    }
    if (P->infinity) { 
        for(int i=0; i<9; i++) R->X.v[i] = Q->point.X.v[i];
        for(int i=0; i<9; i++) R->Y.v[i] = Q->point.Y.v[i];
        for(int i=0; i<9; i++) R->Z.v[i] = Q->point.Z.v[i];
        R->infinity = Q->point.infinity; 
        return; 
    }

    fe29_t z2; fe29_sqr(&z2, &P->Z);
    fe29_t z3; fe29_mul(&z3, &P->Z, &z2);
    
    fe29_t u2; fe29_mul(&u2, &Q->point.X, &z2);
    fe29_t s2; fe29_mul(&s2, &Q->point.Y, &z3);
    
    fe29_t h; fe29_sub(&h, &u2, &P->X);
    fe29_t h_mul2; fe29_mul_int(&h_mul2, &h, 2);
    fe29_t i; fe29_sqr(&i, &h_mul2);
    fe29_t j; fe29_mul(&j, &h, &i);
    
    fe29_t s2_sub_s1; fe29_sub(&s2_sub_s1, &s2, &P->Y);
    fe29_t r; fe29_mul_int(&r, &s2_sub_s1, 2);
    fe29_t v; fe29_mul(&v, &P->X, &i);

    fe29_t r_sqr; fe29_sqr(&r_sqr, &r);
    fe29_sub(&R->X, &r_sqr, &j);
    fe29_t v_mul2; fe29_mul_int(&v_mul2, &v, 2);
    fe29_sub(&R->X, &R->X, &v_mul2);

    fe29_sub(&R->Y, &v, &R->X);
    fe29_mul(&R->Y, &R->Y, &r);
    
    fe29_t s1_j; fe29_mul(&s1_j, &P->Y, &j);
    fe29_t s1_j_mul2; fe29_mul_int(&s1_j_mul2, &s1_j, 2);
    fe29_sub(&R->Y, &R->Y, &s1_j_mul2);

    fe29_mul(&R->Z, &P->Z, &h);
    fe29_mul_int(&R->Z, &R->Z, 2);
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
    uint dp_max_count
) {
    int id = get_global_id(0);
    
    WalkState_cl w;
    w.walk_id = walkers[id].walk_id;
    w.snapshot_steps = walkers[id].snapshot_steps;
    w.state = walkers[id].state;
    for(int i=0; i<4; i++) w.a[i] = walkers[id].a[i];
    for(int i=0; i<4; i++) w.b[i] = walkers[id].b[i];
    for(int i=0; i<4; i++) w.snapshot_x[i] = walkers[id].snapshot_x[i];
    w.R.infinity = walkers[id].R.infinity;
    for(int i=0; i<9; i++) w.R.X.v[i] = walkers[id].R.X.v[i];
    for(int i=0; i<9; i++) w.R.Y.v[i] = walkers[id].R.Y.v[i];
    for(int i=0; i<9; i++) w.R.Z.v[i] = walkers[id].R.Z.v[i];

    for(uint step = 0; step < n_steps_to_run; step++) {
        fe29_t Zinv; fe29_inv(&Zinv, &w.R.Z);
        fe29_t Zinv2; fe29_sqr(&Zinv2, &Zinv);
        fe29_t Zinv3; fe29_mul(&Zinv3, &Zinv2, &Zinv);
        fe29_t Xaff; fe29_mul(&Xaff, &w.R.X, &Zinv2);
        fe29_t Yaff; fe29_mul(&Yaff, &w.R.Y, &Zinv3);
        
        ulong c[9];
        for(int i=0; i<9; i++) c[i] = Xaff.v[i];
        fe29_t Xaff_reduced; fe29_reduce(&Xaff_reduced, c);
        
        ulong X64[4];
        X64[0] = Xaff_reduced.v[0] | ((ulong)Xaff_reduced.v[1] << 29) | (((ulong)Xaff_reduced.v[2] & 0x3F) << 58);
        X64[1] = ((ulong)Xaff_reduced.v[2] >> 6) | ((ulong)Xaff_reduced.v[3] << 23) | (((ulong)Xaff_reduced.v[4] & 0x1FFFF) << 52);
        X64[2] = ((ulong)Xaff_reduced.v[4] >> 17) | ((ulong)Xaff_reduced.v[5] << 12) | (((ulong)Xaff_reduced.v[6] & 0x7FFFFFF) << 41);
        X64[3] = ((ulong)Xaff_reduced.v[6] >> 27) | ((ulong)Xaff_reduced.v[7] << 2) | ((ulong)Xaff_reduced.v[8] << 31);
        
        for(int i=0; i<9; i++) c[i] = Yaff.v[i];
        fe29_t Yaff_reduced; fe29_reduce(&Yaff_reduced, c);
        bool negate = Yaff_reduced.v[0] & 1;
        
        ulong combined = X64[0] ^ (X64[1] << 1) ^ (X64[2] << 2) ^ (X64[3] << 3);
        uint step_idx = murmur_hash3(combined) % step_table_size;
        
        Step_cl step_point;
        step_point.point.infinity = stepTable[step_idx].point.infinity;
        for(int i=0; i<4; i++) step_point.a[i] = stepTable[step_idx].a[i];
        for(int i=0; i<9; i++) step_point.point.X.v[i] = stepTable[step_idx].point.X.v[i];
        for(int i=0; i<9; i++) step_point.point.Y.v[i] = stepTable[step_idx].point.Y.v[i];
        for(int i=0; i<9; i++) step_point.point.Z.v[i] = stepTable[step_idx].point.Z.v[i];
        
        if (negate) {
            fe29_neg(&step_point.point.Y, &step_point.point.Y);
            scalarSub(w.a, step_point.a);
        } else {
            scalarAdd(w.a, step_point.a);
        }
        
        ECPointJacobian_cl new_R;
        jacobianAdd(&new_R, &w.R, &step_point);
        for(int i=0; i<9; i++) w.R.X.v[i] = new_R.X.v[i];
        for(int i=0; i<9; i++) w.R.Y.v[i] = new_R.Y.v[i];
        for(int i=0; i<9; i++) w.R.Z.v[i] = new_R.Z.v[i];
        w.R.infinity = new_R.infinity;

        w.snapshot_steps++;
        
        ulong dp_mask = (1ul << dp_bits) - 1;
        if ((X64[0] & dp_mask) == 0) {
            uint idx = atomic_inc(dp_count);
            if (idx < dp_max_count) {
                for(int k=0; k<4; k++) dp_buffer[idx].a[k] = w.a[k];
                for(int k=0; k<4; k++) dp_buffer[idx].b[k] = w.b[k];
                for(int k=0; k<4; k++) dp_buffer[idx].x[k] = X64[k];
                dp_buffer[idx].walk_id = id;
            }
            w.snapshot_steps = 0;
            w.state = 1; // DP found
            break;
        }
    }
    
    walkers[id].walk_id = w.walk_id;
    walkers[id].snapshot_steps = w.snapshot_steps;
    walkers[id].state = w.state;
    for(int i=0; i<4; i++) walkers[id].a[i] = w.a[i];
    for(int i=0; i<4; i++) walkers[id].b[i] = w.b[i];
    for(int i=0; i<4; i++) walkers[id].snapshot_x[i] = w.snapshot_x[i];
    walkers[id].R.infinity = w.R.infinity;
    for(int i=0; i<9; i++) walkers[id].R.X.v[i] = w.R.X.v[i];
    for(int i=0; i<9; i++) walkers[id].R.Y.v[i] = w.R.Y.v[i];
    for(int i=0; i<9; i++) walkers[id].R.Z.v[i] = w.R.Z.v[i];
}
#endif
