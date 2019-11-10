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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>

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
#ifdef Z_AXIS
	return (c1.x - c2.x) * (c1.x - c2.x) + (c1.y - c2.y) * (c1.y - c2.y) + (c1.z - c2.z) * (c1.z - c2.z);
#else
	return (c1.x - c2.x) * (c1.x - c2.x) + (c1.y - c2.y) * (c1.y - c2.y);
#endif // Z_AXIS

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
#ifdef Z_AXIS
	c3.z = c1.z + m;
#else
	c3.z = c1.z;
#endif // Z_AXIS
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
	if (l > 0.000001)
	{
		cData c1 = empty();
		c1 = multiply(c, m / sqrt(l));
		//printf("%f:%f:%f:::%f:%f:%f:::%f:%f:%f\n", l,getSquaredDistance(c1), m, c.x, c.y, c.z, c1.x, c1.y, c1.z);
		return c1;
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


__global__ void update_k(cData* part, cData* v, 
	int dx, float dt, int* grid_begin, int* grid_end,
	int* ids)
{
	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (gtidx < dx)
	{
		int fj = gtidx;

		cData pterm = part[2*fj];
		cData vterm = v[fj];
		cData fterm = empty();

		int x_grid = int(pterm.x * GRID_SIZE);
		int y_grid = int(pterm.y * GRID_SIZE);
		int z_grid = int(pterm.z * GRID_SIZE);


		//SHORE FUNCTIONS
		cData mid_cohesion = empty();
		cData mid_alignment = empty();
		cData mid_separation = empty();
		cData mid_obstacles = empty();
		int count_cohesion = 0;
		int count_alignment = 0;
		int count_separation = 0;
		
		for (int x = MAX(1, x_grid) - 1; x <= GRID_SIZE && x <= x_grid + 1; x++)
			for (int y = MAX(1, y_grid) - 1; y <= GRID_SIZE && y <= y_grid + 1; y++)
				for (int z = MAX(1, z_grid) - 1; z <= GRID_SIZE && z <= z_grid + 1; z++)
				{
					int grid_id = x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
					for (int id = grid_begin[grid_id]; id < grid_end[grid_id]; id++)
					{
						int i = ids[id];
						if (i != fj && i>=0 && i<=dx)
						{
							float sqr_distance = getSquaredDistance(part[2 * i], pterm);

							//COHESION
							if (sqr_distance < SIGN_RADIUS * SIGN_RADIUS)
							{
								count_cohesion++;
								mid_cohesion = add(mid_cohesion, part[2 * i]);
							}

							//ALIGNMENT
							if (sqr_distance < SIGN_RADIUS * SIGN_RADIUS)
							{
								count_alignment++;
								mid_alignment = add(mid_alignment, v[i]);
							}

							//SEPARATION
							if (sqr_distance < SEPARATION_RADIUS * SEPARATION_RADIUS)
							{
								count_separation++;

								cData tmp = subtract(pterm, part[2 * i]);
								tmp = abs(tmp);
								tmp = multiply(tmp, -1);
								tmp = add(tmp, SEPARATION_RADIUS);

								if (sqr_distance < SEPARATION_RADIUS * SEPARATION_RADIUS / 100)
								{
									tmp = multiply(tmp, 3);
								}
								mid_separation.x += tmp.x * (pterm.x > part[2 * i].x ? 1 : -1);
								mid_separation.y += tmp.y * (pterm.y > part[2 * i].y ? 1 : -1);
								mid_separation.z += tmp.z * (pterm.z > part[2 * i].z ? 1 : -1);
							}
						}
					}
				}

		if (count_cohesion)
		{
			mid_cohesion = divide(mid_cohesion, count_cohesion);
			cData des = subtract(mid_cohesion, pterm);

			cData steer = limit(des, MAX_FORCE);
			steer = multiply(steer, COH_MULTI);

			fterm = add(fterm, steer);
		}
		if (count_alignment)
		{
			mid_alignment = divide(mid_alignment, count_alignment);

			cData steer = setMagnitude(mid_alignment, MAX_FORCE);
			steer = multiply(steer, ALI_MULTI);

			fterm = add(fterm, steer);
		}
		if (count_separation)
		{
			mid_separation = divide(mid_separation, count_separation);

			cData steer = setMagnitude(mid_separation, MAX_FORCE);
			steer = multiply(steer, SEP_MULTI);

			fterm = add(fterm, steer);
		}


		//AVOID OBSTACLES
		float r = OBSTACLES_RADIUS;
		if (pterm.x < 0.25 + r)
		{
			mid_obstacles.x += 0.25 + r - pterm.x;
		}
		if (pterm.y < 0.25 + r)
		{
			mid_obstacles.y += 0.25 + r - pterm.y;
		}
		if (pterm.z < 0.25 + r)
		{
#ifdef Z_AXIS
			mid_obstacles.z += 0.25 + r - pterm.z;
#endif // Z_AXIS
		}
		if (pterm.x > 0.75 - r)
		{
			mid_obstacles.x += (0.75 - pterm.x) - r;
		}
		if (pterm.y > 0.75 - r)
		{
			mid_obstacles.y += (0.75 - pterm.y) - r;
		}
		if (pterm.z > 0.75 - r)
		{
#ifdef Z_AXIS
			mid_obstacles.z += (0.75 - pterm.z) - r;
#endif // Z_AXIS
		}

		mid_obstacles = multiply(mid_obstacles, OBS_MULTI);
		fterm = add(fterm, mid_obstacles);


		//APPLY FORCES
		{
			cData tmp = multiply(fterm, dt);
			vterm = add(vterm, tmp);
			vterm = setMagnitude(vterm, MAX_SPEED);

			v[fj] = vterm;
		}


		//ADVERT PARTICLES
		{
			cData tmp = multiply(vterm, dt);
			tmp = add(tmp, 1.f);
			pterm = add(pterm, tmp);
			pterm.x = pterm.x - (int)pterm.x;
			pterm.y = pterm.y - (int)pterm.y;
			pterm.z = pterm.z - (int)pterm.z;

			float size_back = 0.02 * FISH_SIZE;

			cData v_scalled = setMagnitude(vterm, size_back);

			part[2 * fj] = pterm;
			part[2 * fj + 1] = subtract(pterm, v_scalled);
		}

	}
}

__global__ void get_grid_location_k(cData* part, int* ids, int* grid_ids, int dx)
{
	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (gtidx < dx)
	{
		int fj = gtidx;

		cData pterm = part[2 * fj];

		grid_ids[fj] = int(pterm.x * GRID_SIZE) + int(pterm.y * GRID_SIZE) * GRID_SIZE + int(pterm.z * GRID_SIZE) * GRID_SIZE * GRID_SIZE;
		ids[fj] = fj;
	}
}


__global__ void get_grid_boundries_k(int* grid_ids, int* grid_begin, int* grid_end, int dx)
{
	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (gtidx < dx)
	{
		int fj = gtidx;

		if (fj == 0)
			grid_begin[grid_ids[fj]] = fj;
		if (fj == dx - 1)
			grid_end[grid_ids[fj]] = fj+1;
		else
		{
			if (grid_ids[fj] != grid_ids[fj + 1])
			{
				grid_end[grid_ids[fj]] = fj+1;
				grid_begin[grid_ids[fj + 1]] = fj + 1;
			}
		}
	}
}

extern "C"
void advectParticles(GLuint vbo, cData * v, int* ids, int* grid_ids, int* grid_begin, int* grid_end, int dx, float dt)
{
	dim3 grid(dx/512, 1);
	dim3 tids(512, 1);

	cData* p;
	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	getLastCudaError("cudaGraphicsMapResources failed");

	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void**)&p, &num_bytes,
		cuda_vbo_resource);
	getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");


	get_grid_location_k << <grid, tids >>> (p, ids, grid_ids, dx);
	getLastCudaError("get_grid_location_k failed.");

	thrust::device_ptr<int> keys(grid_ids);
	thrust::device_ptr<int> values(ids);
	thrust::sort_by_key(keys, keys + dx, values);
	getLastCudaError("thrust sorting failed!");

	thrust::device_ptr<int> grid_begin_thrust(grid_begin);
	thrust::device_ptr<int> grid_end_thrust(grid_end);
	thrust::fill(grid_begin_thrust, grid_begin_thrust + GRID_SIZE* GRID_SIZE* GRID_SIZE, -1);
	thrust::fill(grid_end_thrust, grid_end_thrust + GRID_SIZE* GRID_SIZE* GRID_SIZE, -1);
	getLastCudaError("thrust sorting failed!");

	get_grid_boundries_k <<< grid, tids >>> (grid_ids, grid_begin, grid_end, GRID_SIZE * GRID_SIZE * GRID_SIZE);

	update_k << < grid, tids >> > (p, v, dx, dt, grid_begin, grid_end, ids);
	getLastCudaError("update_k failed.");


	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
	getLastCudaError("cudaGraphicsUnmapResources failed");
}



extern "C"
void test(GLuint vbo, cData * v, int dx, float dt)
{
	float sum_milliseconds = 0;
	int times = 100;

	for (int i = 0; i < times; i++)
	{
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		//advectParticles(vbo, v, dx, dt);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		sum_milliseconds += milliseconds;
	}
	
	printf("Avarage time of update: %f", sum_milliseconds/times);
}