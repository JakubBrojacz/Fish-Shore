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
 
#ifndef DEFINES_H
#define DEFINES_H

#define Z_AXIS
#define TEST_TIME false


#define SHORE 512
#define SHORE_ARR (SHORE * 2)	// each fish consists of 2 points

#define DT     0.09f     // Delta T is constant

#define FISH_SIZE 0.5f

#define SIGN_RADIUS	(0.04f * FISH_SIZE)
#define SEPARATION_RADIUS (0.02f * FISH_SIZE)
#define OBSTACLES_RADIUS 0.025f

#define MAX_SPEED 0.01f
#define MAX_FORCE 0.0015f

#define SEP_MULTI 3.f
#define ALI_MULTI 2.f
#define COH_MULTI 1.f
#define OBS_MULTI 2.f


#define GRID_SIZE (1.0f/SIGN_RADIUS)

#endif
