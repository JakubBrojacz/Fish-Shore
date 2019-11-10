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


#define SHORE 1024
#define SHORE_ARR (SHORE * 6)	// each fish consists of 6 points

#define DT     0.09f     // Delta T is constant

#define FISH_SIZE 0.5

#define SIGN_RADIUS	(0.04 * FISH_SIZE)
#define SEPARATION_RADIUS (0.02 * FISH_SIZE)
#define OBSTACLES_RADIUS 0.025

#define MAX_SPEED 0.01
#define MAX_FORCE 0.0015

#define SEP_MULTI 3
#define ALI_MULTI 2
#define COH_MULTI 1
#define OBS_MULTI 2


#define GRID_SIZE (1/SIGN_RADIUS)

#endif
