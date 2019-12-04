#ifndef DEFINES_H
#define DEFINES_H

#define Z_AXIS
#define TEST_TIME false


#define SHORE (4096*4)
 // each fish consists of 2 points
#define SHORE_ARR (SHORE * 2)

// Delta T is constant
#define DT 0.09f

#define FISH_SIZE 0.2f

#define SIGN_RADIUS	(0.04f * FISH_SIZE)
#define SEPARATION_RADIUS (0.02f * FISH_SIZE)
#define OBSTACLES_RADIUS 0.025f

#define MAX_SPEED 0.01f
#define MAX_FORCE 0.0015f

//how much seperate values affect fish
#define SEP_MULTI 3.f
#define ALI_MULTI 5.f
#define COH_MULTI 1.f
#define OBS_MULTI 2.f


#define MIN_DIM 0.25
#define MAX_DIM 0.75

#define GRID_SIZE ((int)((MAX_DIM-MIN_DIM)/SIGN_RADIUS))

#endif
