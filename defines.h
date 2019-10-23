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

#define SHORE 500
#define SHORE_ARR (SHORE * 6)

#define DT     0.09f     // Delta T for interative solver
#define VIS    0.0025f   // Viscosity constant
#define FORCE (5.8f*DIM) // Force scale factor 
#define FR     4         // Force update radius

#define FISH_SIZE 1

#define SIGN_RADIUS	(0.04 * FISH_SIZE)
#define SEPARATION_RADIUS (0.02 * FISH_SIZE)
#define OBSTACLES_RADIUS 0.05

#define MAX_SPEED 0.01 
#define MAX_FORCE 0.0015

#define SEP_MULTI 1
#define ALI_MULTI 1
#define COH_MULTI 1
#define OBS_MULTI 2

#endif
