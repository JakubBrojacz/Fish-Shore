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

#define SHORE 10
#define SHORE_ARR (SHORE * 6)

#define DT     0.09f     // Delta T for interative solver
#define VIS    0.0025f   // Viscosity constant
#define FORCE (5.8f*DIM) // Force scale factor 
#define FR     4         // Force update radius

#define TILEX 64 // Tile width
#define TILEY 64 // Tile height
#define TIDSX 64 // Tids in X
#define TIDSY 4  // Tids in Y

#endif
