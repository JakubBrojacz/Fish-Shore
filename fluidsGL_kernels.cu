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


__device__ inline float getSquaredDistance(cData c1, cData c2)
{
	return (c1.x - c2.x) * (c1.x - c2.x) + (c1.y - c2.y) * (c1.y - c2.y);
}

__device__ cData setMagnitude(cData c, float m)
{
	float l = (c.x * c.x) + (c.y * c.y);
	if (l > 0.0001)
	{
		c.x = c.x * (m * m) / l;
		c.y = c.y * (m * m) / l;
	}
	return c;
}

__device__ cData limit(cData c, float m)
{
	float l = (c.x * c.x) + (c.y * c.y);
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
		float midx = 0, midy = 0;
		for (int i = 0; i < SHORE; i++)
			if (i != fj && getSquaredDistance(part[6 * i], pterm) < SIGN_RADIUS * SIGN_RADIUS)
			{
				count++;
				midx += part[i].x;
				midy += part[i].y;
			}
		if (count > 0)
		{
			midx /= count;
			midy /= count;

			float des_x = midx - pterm.x;
			float des_y = midy - pterm.y;

			cData des = cData();
			des.x = des_x;
			des.y = des_y;
			des = setMagnitude(des, MAX_SPEED);

			cData steer = cData();
			/*steer.x = (des_x - v[fj].x);
			steer.y = (des_y - v[fj].y);*/
			steer.x = (des_x);
			steer.y = (des_y);
			steer = limit(steer, MAX_FORCE);

			f[fj].x += steer.x * COH_MULTI;
			f[fj].y += steer.y * COH_MULTI;
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
		float midx = 0, midy = 0;
		for (int i = 0; i < SHORE; i++)
			if (i != fj && getSquaredDistance(part[6 * i], pterm) < SEPARATION_RADIUS * SEPARATION_RADIUS)
			{
				count++;
				cData tmp = cData();
				tmp.x = SEPARATION_RADIUS - abs(pterm.x - part[6 * i].x);
				tmp.y = SEPARATION_RADIUS - abs(pterm.y - part[6 * i].y);
				//tmp = setMagnitude(tmp, 1 / sqrt(getSquaredDistance(part[6 * i], pterm)));
				midx += tmp.x * (pterm.x > part[6 * i].x ? 1 : -1);
				midy += tmp.y * (pterm.y > part[6 * i].y ? 1 : -1);
				if (getSquaredDistance(part[6 * i], pterm) < SEPARATION_RADIUS * SEPARATION_RADIUS / 100)
				{
					tmp.x = tmp.x * tmp.x - SEPARATION_RADIUS * SEPARATION_RADIUS * 81 / 100;
					tmp.y = tmp.y * tmp.y - SEPARATION_RADIUS * SEPARATION_RADIUS * 81 / 100;
					midx += tmp.x * (pterm.x > part[6 * i].x ? 1 : -1);
					midy += tmp.y * (pterm.y > part[6 * i].y ? 1 : -1);
				}
			}
		if (count > 0)
		{
			midx /= count;
			midy /= count;

			cData steer = cData();
			/*steer.x = (des_x - v[fj].x);
			steer.y = (des_y - v[fj].y);*/
			steer.x = (midx);
			steer.y = (midy);
			steer = setMagnitude(steer, MAX_FORCE);

			f[fj].x += steer.x * SEP_MULTI;
			f[fj].y += steer.y * SEP_MULTI;
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
		float midx = 0, midy = 0;
		for (int i = 0; i < SHORE; i++)
			if (i != fj && getSquaredDistance(part[6 * i], pterm) < SIGN_RADIUS * SIGN_RADIUS)
			{
				count++;
				midx += v[i].x;
				midy += v[i].y;
			}
		if (count > 0)
		{
			midx /= count;
			midy /= count;

			float des_x = midx;
			float des_y = midy;

			cData des = cData();
			des.x = des_x;
			des.y = des_y;
			des = setMagnitude(des, MAX_SPEED);

			cData steer = cData();
			/*steer.x = (des_x - v[fj].x);
			steer.y = (des_y - v[fj].y);*/
			steer.x = (des_x);
			steer.y = (des_y);
			steer = limit(steer, MAX_FORCE);

			f[fj].x += steer.x * ALI_MULTI;
			f[fj].y += steer.y * ALI_MULTI;
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
			cData mid = cData();
			mid.x = 0;
			mid.y = 0;
			if (pterm.x < r)
			{
				mid.x += r - pterm.x;
			}
			if (pterm.y < r)
			{
				mid.y += r - pterm.y;
			}
			if (pterm.x > 1 - r)
			{
				mid.x += (1 - pterm.x) - r;
			}
			if (pterm.y > 1 - r)
			{
				mid.y += (1 - pterm.y) - r;
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

			fterm.x += mid.x * OBS_MULTI;
			fterm.y += mid.y * OBS_MULTI;

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

			vterm.x += dt * fterm.x;
			vterm.y += dt * fterm.y;
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

			pterm.x += dt * vterm.x;
			pterm.x = pterm.x - (int)pterm.x;
			pterm.x += 1.f;
			pterm.x = pterm.x - (int)pterm.x;
			pterm.y += dt * vterm.y;
			pterm.y = pterm.y - (int)pterm.y;
			pterm.y += 1.f;
			pterm.y = pterm.y - (int)pterm.y;

			float vx = vterm.x;
			float vy = vterm.y;
			alpha[fj] = atan2(-vy, -vx);

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
