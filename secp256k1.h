/***********************************************************************************************************
* This file is part of the Pollard's Lambda distribution: (https://github.com/lucaselblanc/pollardslambda) *
* Copyright (c) 2024, 2026 Lucas Leblanc.                                                                  *
* Distributed under the MIT software license, see the accompanying.                                        *
* file COPYING or https://www.opensource.org/licenses/mit-license.php.                                     *
************************************************************************************************************/

/*******************************************
* Pollard's Lambda Algorithm for SECP256K1 *
* Written by Lucas Leblanc                 *
********************************************/

#ifndef EC_SECP256K1_H
#define EC_SECP256K1_H

#include "hip_utils.h"


#include <stdint.h>
#include "parallel_hashmap/phmap.h"
#include <condition_variable>
#include <openssl/sha.h>
#include <fstream>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <limits>
#include <climits>
#include <ctime>
#include <cmath>
#include <cstring>
#include <tuple>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

struct uint256_t {
    uint64_t limbs[4];
};



typedef struct {
    uint64_t x[4];
    uint64_t y[4];
    int infinity;
} ECPointAffine;

typedef struct {
    uint64_t X[4];
    uint64_t Y[4];
    uint64_t Z[4];
    int infinity;
} ECPointJacobian;

using uint128_t = unsigned __int128;

extern ECPointJacobian* preCompG;
extern ECPointJacobian* preCompGphi;
extern ECPointJacobian* preCompH;
extern ECPointJacobian* preCompHphi;
extern ECPointJacobian* jacNorm;
extern ECPointJacobian* jacEndo;
extern ECPointJacobian* jacNormH;
extern ECPointJacobian* jacEndoH;

#ifndef HD
#define HD __host__ __device__
#endif

uint256_t modinv(uint256_t base, uint256_t mod);

HD void affineToJacobian(ECPointJacobian *jac, const ECPointAffine *aff);
void decompressPublicKey(ECPointAffine* out, const unsigned char compressed[33]);
HD void endomorphismMap(ECPointJacobian *R, const ECPointJacobian *P);
HD void fromMontgomeryP(uint64_t *result, const uint64_t *a);
void generatePublicKey(ECPointJacobian *preCompTable, ECPointJacobian *preCompTablePhi, unsigned char *out, const uint64_t *PRIV_KEY, int windowSize);
void initPreCompG(int windowSize);
void initPreCompH(const ECPointJacobian *h, int windowSize);
HD void jacobianScalarMultPhi(ECPointJacobian *result, ECPointJacobian *preCompTable, ECPointJacobian *preCompTablePhi, const uint64_t *scalar, int windowSize);
HD void jacobianDouble(ECPointJacobian *R, const ECPointJacobian *P);
HD void jacobianAdd(ECPointJacobian *R, const ECPointJacobian *P, const ECPointJacobian *Q);
HD void jacobianToAffine(ECPointAffine *aff, const ECPointJacobian *jac);
HD void jacobianSetInfinity(ECPointJacobian *point);
HD bool jacobianIsInfinity(const ECPointJacobian *P);
HD void modMulMontP(uint64_t *result, const uint64_t *a, const uint64_t *b);
HD void modSubP(uint64_t *result, const uint64_t *a, const uint64_t *b);
HD void modAddP(uint64_t *result, const uint64_t *a, const uint64_t *b);
HD void modExpMontP(uint64_t *res, const uint64_t *base, const uint64_t *exp);
HD void pointInitJacobian(ECPointJacobian *P);
HD void pointAddJacobian(ECPointJacobian *R, const ECPointJacobian *P, const ECPointJacobian *Q);
HD void pointDoubleJacobian(ECPointJacobian *R, const ECPointJacobian *P);
HD void scalarReduceN(uint64_t *r, const uint64_t *k);
HD void scalarMul(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]);
HD void scalarAdd(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]);
HD void scalarSub(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]);
HD void scalarNeg(uint64_t r[4], const uint64_t a[4]);
HD int scalarIsZero(const uint64_t a[4]);
void serializePublicKey(unsigned char *out, const ECPointAffine *publicKey);
HD void toMontgomeryP(uint64_t *result, const uint64_t *a);

#endif /* EC_SECP256K1_H */
