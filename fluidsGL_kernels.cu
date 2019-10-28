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

// Particle data
extern GLuint vbo;                 // OpenGL vertex buffer object
extern struct cudaGraphicsResource* cuda_vbo_resource; // handles OpenGL-CUDA exchange


__device__ inline cData empty()
{
	cData tmp = cData();
	tmp.x = tmp.y = tmp.z = 0;
	return tmp;
}


__device__ inline float getSquaredDistance(cData c1, cData c2=empty())
{
	return (c1.x - c2.x) * (c1.x - c2.x) + (c1.y - c2.y) * (c1.y - c2.y) + (c1.z - c2.z) * (c1.z - c2.z);
}

__device__ inline cData add(cData c1, cData c2)
{
	cData c3 = empty();
	c3.x = c1.x + c2.x;
	c3.y = c1.y + c2.y;
	c3.z = c1.z + c2.z;
	return c3;
}

__device__ inline cData add(cData c1, float m)
{
	cData c3 = empty();
	c3.x = c1.x + m;
	c3.y = c1.y + m;
	c3.z = c1.z + m;
	return c3;
}

__device__ inline cData subtract(cData c1, cData c2)
{
	cData c3 = empty();
	c3.x = c1.x - c2.x;
	c3.y = c1.y - c2.y;
	c3.z = c1.z - c2.z;
	return c3;
}

__device__ inline cData multiply(cData c1, cData c2)
{
	cData c3 = empty();
	c3.x = c1.x * c2.x;
	c3.y = c1.y * c2.y;
	c3.z = c1.z * c2.z;
	return c3;
}

__device__ inline cData multiply(cData c1, float m)
{
	cData c3 = empty();
	c3.x = c1.x * m;
	c3.y = c1.y * m;
	c3.z = c1.z * m;
	return c3;
}

__device__ inline cData divide(cData c1, float m)
{
	cData c3 = empty();
	c3.x = c1.x / m;
	c3.y = c1.y / m;
	c3.z = c1.z / m;
	return c3;
}

__device__ inline cData abs(cData c1)
{
	cData c3 = empty();
	c3.x = abs(c1.x);
	c3.y = abs(c1.y);
	c3.z = abs(c1.z);
	return c3;
}

__device__ cData setMagnitude(cData c, float m)
{
	float l = getSquaredDistance(c);
	if (l > 0.0001)
	{
		c = multiply(c, m * m / l);
	}
	return c;
}

__device__ cData limit(cData c, float m)
{
	float l = getSquaredDistance(c);
	if (l > m* m)
	{
		return setMagnitude(c, m);
	}
	return c;
}


__global__ void
prepare_k(cData* f, int dx, int dy,
	float dt, int lb)
{
	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;

	if (gtidx < dx && gtidy < dy)
	{
		int fj = gtidx + gtidy * dx;
		f[fj].x = 0;
		f[fj].y = 0;
		f[fj].z = 0;
	}
}


__global__ void
cohesion_k(cData* part, cData* v, cData* f, int dx, int dy,
	float dt, int lb)
{
	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;

	cData pterm;

	if (gtidx < dx && gtidy < dy)
	{
		int fj = gtidx + gtidy * dx;
		pterm = part[6 * fj];

		int count = 0;
		cData mid = empty();
		for (int i = 0; i < SHORE; i++)
			if (i != fj && getSquaredDistance(part[6 * i], pterm) < SIGN_RADIUS * SIGN_RADIUS)
			{
				count++;
				mid = add(mid, part[6*i]);
			}
		if (count > 0)
		{
			mid = divide(mid, count);
			cData des = subtract(mid, pterm);

			cData steer = limit(des, MAX_FORCE);
			steer = multiply(steer, COH_MULTI);

			f[fj] = add(f[fj], steer);
		}
	}
}

__global__ void
separation_k(cData* part, cData* v, cData* f, int dx, int dy,
	float dt, int lb)
{
	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;

	cData pterm;

	if (gtidx < dx && gtidy < dy)
	{
		int fj = gtidx + gtidy * dx;
		pterm = part[6 * fj];

		int count = 0;
		cData mid = empty();
		for (int i = 0; i < SHORE; i++)
			if (i != fj && getSquaredDistance(part[6 * i], pterm) < SEPARATION_RADIUS * SEPARATION_RADIUS)
			{
				count++;

				//tmp = SEPARATION_RADIUS - abs(pterm - part[6 * i])
				cData tmp = subtract(pterm, part[6 * i]);
				tmp = abs(tmp);
				tmp = multiply(tmp, -1);
				tmp = add(tmp, SEPARATION_RADIUS);

			/*	if (getSquaredDistance(part[6 * i], pterm) < SEPARATION_RADIUS * SEPARATION_RADIUS / 100)
				{
					tmp = multiply(tmp, tmp);
				}*/
				mid.x += tmp.x * (pterm.x > part[6 * i].x ? 1 : -1);
				mid.y += tmp.y * (pterm.y > part[6 * i].y ? 1 : -1);
				mid.z += tmp.z * (pterm.z > part[6 * i].z ? 1 : -1);
			}
		if (count > 0)
		{
			mid = divide(mid, count);

			cData steer = setMagnitude(mid, MAX_FORCE);
			steer = multiply(steer, SEP_MULTI);

			f[fj] = add(f[fj], steer);
		}
	}
}


__global__ void
alignment_k(cData* part, cData* v, cData* f, int dx, int dy,
	float dt, int lb)
{
	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;

	cData pterm;

	if (gtidx < dx && gtidy < dy)
	{
		int fj = gtidx + gtidy * dx;
		pterm = part[6 * fj];

		int count = 0;
		cData mid = empty();
		for (int i = 0; i < SHORE; i++)
			if (i != fj && getSquaredDistance(part[6 * i], pterm) < SIGN_RADIUS * SIGN_RADIUS)
			{
				count++;
				mid = add(mid, v[i]);
			}
		if (count > 0)
		{
			mid = divide(mid, count);

			cData steer = setMagnitude(mid, MAX_FORCE);
			steer = multiply(steer, ALI_MULTI);

			f[fj] = add(f[fj], steer);
		}
	}
}


__global__ void
avoidEdges_k(cData* p, cData* v, cData* f, int dx, int dy,
	float dt, int lb)
{

	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;

	// gtidx is the domain location in x for this thread
	cData pterm, fterm;

	if (gtidx < dx)
	{
		if (gtidy < dy)
		{
			int fj = gtidx + gtidy * dx;

			pterm = p[6 * fj];
			fterm = f[fj];


			float r = OBSTACLES_RADIUS;
			cData mid = empty();
			if (pterm.x < r)
			{
				mid.x += r - pterm.x;
			}
			if (pterm.y < r)
			{
				mid.y += r - pterm.y;
			}
			if (pterm.z < r)
			{
				mid.z += r - pterm.z;
			}
			if (pterm.x > 1 - r)
			{
				mid.x += (1 - pterm.x) - r;
			}
			if (pterm.y > 1 - r)
			{
				mid.y += (1 - pterm.y) - r;
			}
			if (pterm.z > 1 - r)
			{
				mid.z += (1 - pterm.z) - r;
			}

			//round vertices
			if (mid.x != 0 && mid.y != 0)
			{
				float addx = abs(mid.y) * (mid.x > 0 ? 1 : -1);
				float addy = abs(mid.x) * (mid.y > 0 ? 1 : -1);
				mid.x += addx;
				mid.y += addy;
			}

			//mid = limit(mid, MAX_FORCE);

			mid = multiply(mid, OBS_MULTI);
			fterm = add(fterm, mid);

			f[fj] = fterm;
		} // If this thread is inside the domain in Y
	} // If this thread is inside the domain in X
}


__global__ void
applyForces_k(cData* v, cData* f, int dx, int dy,
	float dt, int lb)
{

	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;

	// gtidx is the domain location in x for this thread
	cData vterm, fterm;

	if (gtidx < dx)
	{
		if (gtidy < dy)
		{
			int fj = gtidx + gtidy * dx;

			vterm = v[fj];
			fterm = f[fj];

			cData tmp = multiply(fterm, dt);
			vterm = add(vterm, tmp);
			vterm = setMagnitude(vterm, MAX_SPEED);

			v[fj] = vterm;
		} // If this thread is inside the domain in Y
	} // If this thread is inside the domain in X
}


// This method updates the particles by moving particle positions
// according to the velocity field and time step. That is, for each
// particle: p(t+1) = p(t) + dt * v(p(t)).
__global__ void
advectParticles_k(cData* part, cData* v, float* alpha, int dx, int dy,
	float dt, int lb)
{

	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;

	// gtidx is the domain location in x for this thread
	cData pterm, vterm;

	if (gtidx < dx)
	{
		if (gtidy < dy)
		{
			int fj = gtidx + gtidy * dx;
			pterm = part[6 * fj];


			/*v[fj].y = -sin(alpha[fj]) * 0.01;
			v[fj].x = -cos(alpha[fj]) * 0.01;*/


			vterm = v[fj];

			cData tmp = multiply(vterm, dt);
			tmp = add(tmp, 1.f);
			pterm = add(pterm, tmp);
			pterm.x = pterm.x - (int)pterm.x;
			pterm.y = pterm.y - (int)pterm.y;
			pterm.z = pterm.z - (int)pterm.z;

			alpha[fj] = atan2(-vterm.y, -vterm.x);

			float size_back = 0.02 * FISH_SIZE;
			float size_fin = 0.01 * FISH_SIZE;

			part[6 * fj] = pterm;
			part[6 * fj + 1].x = pterm.x + cos(alpha[fj]) * size_back;
			part[6 * fj + 1].y = pterm.y + sin(alpha[fj]) * size_back;
			part[6 * fj + 2].x = pterm.x;
			part[6 * fj + 2].y = pterm.y;
			part[6 * fj + 3].x = pterm.x + cos(alpha[fj] + M_PI / 6) * size_fin;
			part[6 * fj + 3].y = pterm.y + sin(alpha[fj] + M_PI / 6) * size_fin;
			part[6 * fj + 4].x = pterm.x;
			part[6 * fj + 4].y = pterm.y;
			part[6 * fj + 5].x = pterm.x + cos(alpha[fj] - M_PI / 6) * size_fin;
			part[6 * fj + 5].y = pterm.y + sin(alpha[fj] - M_PI / 6) * size_fin;
		/*	if (fj == 0)
				printf("%f:%f:%f\n", pterm.x, pterm.y, pterm.z);*/
		} // If this thread is inside the domain in Y
	} // If this thread is inside the domain in X
}

extern "C"
void advectParticles(GLuint vbo, cData * v, cData * f, float* alpha, int dx, int dy, float dt)
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

	prepare_k << < grid, tids >> > (f, SHORE, 1, dt, 1);
	cohesion_k << < grid, tids >> > (p, v, f, SHORE, 1, dt, 1);
	separation_k << < grid, tids >> > (p, v, f, SHORE, 1, dt, 1);
	alignment_k << < grid, tids >> > (p, v, f, SHORE, 1, dt, 1);
	avoidEdges_k << <grid, tids >> > (p, v, f, SHORE, 1, dt, 1);
	applyForces_k << <grid, tids >> > (v, f, SHORE, 1, dt, 1);
	advectParticles_k << < grid, tids >> > (p, v, alpha, SHORE, 1, dt, 1);
	getLastCudaError("advectParticles_k failed.");

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
	getLastCudaError("cudaGraphicsUnmapResources failed");
}
