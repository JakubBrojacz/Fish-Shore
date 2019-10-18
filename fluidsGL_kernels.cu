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

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cufft.h>          // CUDA FFT Libraries
#include <helper_cuda.h>    // Helper functions for CUDA Error handling

 // OpenGL Graphics includes
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>


// FluidsGL CUDA kernel definitions
#include "fluidsGL_kernels.cuh"


#define M_PI 3.14159265

// Texture object for reading velocity field
cudaTextureObject_t     texObj;
static cudaArray* array = NULL;

// Particle data
extern GLuint vbo;                 // OpenGL vertex buffer object
extern struct cudaGraphicsResource* cuda_vbo_resource; // handles OpenGL-CUDA exchange

// Texture pitch
extern size_t tPitch;
extern cufftHandle planr2c;
extern cufftHandle planc2r;
cData* vxfield = NULL;
cData* vyfield = NULL;

void setupTexture(int x, int y)
{
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();

	cudaMallocArray(&array, &desc, y, x);
	getLastCudaError("cudaMalloc failed");

	cudaResourceDesc            texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = array;

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = false;
	texDescr.filterMode = cudaFilterModeLinear;
	texDescr.addressMode[0] = cudaAddressModeWrap;
	texDescr.readMode = cudaReadModeElementType;

	checkCudaErrors(cudaCreateTextureObject(&texObj, &texRes, &texDescr, NULL));
}

void deleteTexture(void)
{
	checkCudaErrors(cudaDestroyTextureObject(texObj));
	checkCudaErrors(cudaFreeArray(array));
}


__device__ inline float getSquaredDistance(cData c1, cData c2)
{
	return (c1.x - c2.x) * (c1.x - c2.x) + (c1.y - c2.y) * (c1.y - c2.y);
}




__global__ void
cohesion_k(cData* part, cData* v, float* alpha, int dx, int dy,
	float dt, int lb, size_t pitch)
{
	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
	int p;

	cData pterm;

	if (gtidx < dx && gtidy < dy)
	{
		int fj = gtidx + gtidy * dx;
		pterm = part[6 * fj];

		int count = 0;
		float midx = 0, midy = 0;
		for (int i = 0; i < SHORE_ARR; i += 6)
			if (i != 6 * fj && getSquaredDistance(part[i], pterm) < SIGN_RADIUS * SIGN_RADIUS)
			{
				count++;
				midx += part[i].x;
				midy += part[i].y;
			}
		midx /= count;
		midy /= count;
		cData midpoint = cData();
		midpoint.x = midx;
		midpoint.y = midy;

		float alpha1 = acos((midx - pterm.x) / sqrt(getSquaredDistance(midpoint, pterm)));
		alpha[fj] += (alpha1 > alpha[fj]) * 0.1;
	}
}

// This method updates the particles by moving particle positions
// according to the velocity field and time step. That is, for each
// particle: p(t+1) = p(t) + dt * v(p(t)).
__global__ void
advectParticles_k(cData* part, cData* v, float* alpha, int dx, int dy,
	float dt, int lb, size_t pitch)
{

	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
	int p;

	// gtidx is the domain location in x for this thread
	cData pterm, vterm;

	if (gtidx < dx)
	{
		if (gtidy < dy)
		{
			int fj = gtidx + gtidy * dx;
			pterm = part[6 * fj];

			
			v[fj].y = -sin(alpha[fj]) * 0.01;
			v[fj].x = -cos(alpha[fj]) * 0.01;


			vterm = v[fj];

			pterm.x += dt * vterm.x;
			pterm.x = pterm.x - (int)pterm.x;
			pterm.x += 1.f;
			pterm.x = pterm.x - (int)pterm.x;
			pterm.y += dt * vterm.y;
			pterm.y = pterm.y - (int)pterm.y;
			pterm.y += 1.f;
			pterm.y = pterm.y - (int)pterm.y;

			part[6 * fj] = pterm;
			part[6 * fj + 1].x = pterm.x + cos(alpha[fj]) * 0.02;
			part[6 * fj + 1].y = pterm.y + sin(alpha[fj]) * 0.02;
			part[6 * fj + 2].x = pterm.x;
			part[6 * fj + 2].y = pterm.y;
			part[6 * fj + 3].x = pterm.x + cos(alpha[fj] + M_PI / 6) * 0.01;
			part[6 * fj + 3].y = pterm.y + sin(alpha[fj] + M_PI / 6) * 0.01;
			part[6 * fj + 4].x = pterm.x;
			part[6 * fj + 4].y = pterm.y;
			part[6 * fj + 5].x = pterm.x + cos(alpha[fj] - M_PI / 6) * 0.01;
			part[6 * fj + 5].y = pterm.y + sin(alpha[fj] - M_PI / 6) * 0.01;
		} // If this thread is inside the domain in Y
	} // If this thread is inside the domain in X
}

extern "C"
void advectParticles(GLuint vbo, cData * v, float* alpha, int dx, int dy, float dt)
{
	dim3 grid(1, 1);
	dim3 tids(SHORE, 1);

	cData* p;
	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	getLastCudaError("cudaGraphicsMapResources failed");

	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void**)&p, &num_bytes,
		cuda_vbo_resource);
	getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

	//cohesion_k << < grid, tids >> > (p, v, alpha, SHORE, 1, dt, 1, tPitch);
	//applyForces_k <<<grid, tids>>> (v, f, SHORE, 1, dt, 1, tPitch)
	advectParticles_k << < grid, tids >> > (p, v, alpha, SHORE, 1, dt, 1, tPitch);
	//getLastCudaError("advectParticles_k failed.");

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
	getLastCudaError("cudaGraphicsUnmapResources failed");
}
