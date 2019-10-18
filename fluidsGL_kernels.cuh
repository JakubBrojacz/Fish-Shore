/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#ifndef __STABLEFLUIDS_KERNELS_CUH_
#define __STABLEFLUIDS_KERNELS_CUH_

#include "defines.h"

// Vector data type used to velocity and force fields
typedef float2 cData;


__device__ inline float getSquaredDistance(cData c1, cData c2);

__global__ void
advectParticles_k(cData *part, cData *v, int dx, int dy,
                  float dt, int lb, size_t pitch);

#endif

