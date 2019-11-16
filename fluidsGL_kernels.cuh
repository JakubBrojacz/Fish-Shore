#ifndef __STABLEFLUIDS_KERNELS_CUH_
#define __STABLEFLUIDS_KERNELS_CUH_

#include "defines.h"

// Vector data type used to velocity and force fields
typedef float3 cData;


__device__ inline float getSquaredDistance(cData c1, cData c2);

__global__ void
update_k(cData* part, cData* v,
	int dx, float dt, int* grid_begin, int* grid_end);

#endif

