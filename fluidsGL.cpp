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

 // OpenGL Graphics includes
#include <helper_gl.h>

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA FFT Libraries
#include <cufft.h>

// CUDA helper functions
#include <helper_functions.h>
#include <rendercheck_gl.h>
#include <helper_cuda.h>

#include "defines.h"
#include "fluidsGL_kernels.h"

#define MAX_EPSILON_ERROR 1.0f

const char* sSDKname = "fluidsGL";
// CUDA example code that implements the frequency space version of
// Jos Stam's paper 'Stable Fluids' in 2D. This application uses the
// CUDA FFT library (CUFFT) to perform velocity diffusion and to
// force non-divergence in the velocity field at each time step. It uses
// CUDA-OpenGL interoperability to update the particle field directly
// instead of doing a copy to system memory before drawing. Texture is
// used for automatic bilinear interpolation at the velocity advection step.

void cleanup(void);
void reshape(int x, int y);

// CUFFT plan handle
cufftHandle planr2c;
cufftHandle planc2r;
static cData* vxfield = NULL;
static cData* vyfield = NULL;

cData* hvfield = NULL;
cData* dvfield = NULL;
static int wWidth = 512;
static int wHeight = 512;

float* halphafield = NULL;
float* dalphafield = NULL;

cData* hffield = NULL;
cData* dffield = NULL;

static int clicked = 0;
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface* timer = NULL;

// Particle data
GLuint vbo = 0;                 // OpenGL vertex buffer object
struct cudaGraphicsResource* cuda_vbo_resource; // handles OpenGL-CUDA exchange
static cData* particles = NULL; // particle positions in host memory
static int lastx = 0, lasty = 0;

// Texture pitch
size_t tPitch = 0; // Now this is compatible with gcc in 64-bit

char* ref_file = NULL;
bool g_bQAAddTestForce = true;
int  g_iFrameToCompare = 100;
int  g_TotalErrors = 0;

bool g_bExitESC = false;

// CheckFBO/BackBuffer class objects
CheckRender* g_CheckRender = NULL;

extern "C" void advectParticles(GLuint vbo, cData * v, cData * f, float* alpha, int dx, int dy, float dt);


void simulateFluids(void)
{
	advectParticles(vbo, dvfield, dffield, dalphafield, SHORE_ARR, 1, DT);
}

void display(void)
{

	if (!ref_file)
	{
		sdkStartTimer(&timer);
		simulateFluids();
	}

	// render points from vertex buffer
	glClear(GL_COLOR_BUFFER_BIT);
	glColor4f(0, 1, 0, 0.5f);
	glPointSize(1);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnableClientState(GL_VERTEX_ARRAY);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(2, GL_FLOAT, 0, NULL);
	glDrawArrays(GL_LINES, 0, SHORE_ARR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisable(GL_TEXTURE_2D);

	if (ref_file)
	{
		return;
	}

	// Finish timing before swap buffers to avoid refresh sync
	sdkStopTimer(&timer);
	glutSwapBuffers();

	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		char fps[256];
		float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		sprintf(fps, "Cuda/GL Stable Fluids (%d x %d): %3.1f fps", SHORE, 1, ifps);
		glutSetWindowTitle(fps);
		fpsCount = 0;
		fpsLimit = (int)MAX(ifps, 1.f);
		sdkResetTimer(&timer);
	}

	glutPostRedisplay();
}

// very simple von neumann middle-square prng.  can't use rand() in -qatest
// mode because its implementation varies across platforms which makes testing
// for consistency in the important parts of this program difficult.
float myrand(void)
{
	static int seed = 72191;
	char sq[22];

	if (ref_file)
	{
		seed *= seed;
		sprintf(sq, "%010d", seed);
		// pull the middle 5 digits out of sq
		sq[8] = 0;
		seed = atoi(&sq[3]);

		return seed / 99999.f;
	}
	else
	{
		return rand() / (float)RAND_MAX;
	}
}

void initParticles(cData* p, int shore_count)
{
	for (int i = 0; i < shore_count; i++)
	{
		p[6 * i].x = (myrand());
		p[6 * i].y = (myrand());
		p[6 * i + 1].x = p[6 * i].x + 0.02;
		p[6 * i + 1].y = p[6 * i].y + 0.02;
		p[6 * i + 2].x = p[6 * i].x;
		p[6 * i + 2].y = p[6 * i].y;
		p[6 * i + 3].x = p[6 * i].x + 0.01;
		p[6 * i + 3].y = p[6 * i].y;
		p[6 * i + 4].x = p[6 * i].x;
		p[6 * i + 4].y = p[6 * i].y;
		p[6 * i + 5].x = p[6 * i].x;
		p[6 * i + 5].y = p[6 * i].y + 0.01;
	}
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:
		g_bExitESC = true;
#if defined (__APPLE__) || defined(MACOSX)
		exit(EXIT_SUCCESS);
#else
		glutDestroyWindow(glutGetWindow());
		return;
#endif
		break;

	default:
		break;
	}
}

void click(int button, int updown, int x, int y)
{
	lastx = x;
	lasty = y;
	clicked = !clicked;
}

void motion(int x, int y)
{
	glutPostRedisplay();
}

void reshape(int x, int y)
{
	wWidth = x;
	wHeight = y;
	glViewport(0, 0, x, y);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 1, 0, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glutPostRedisplay();
}

void cleanup(void)
{
	cudaGraphicsUnregisterResource(cuda_vbo_resource);

	// Free all host and device resources
	free(hvfield);
	free(hffield);
	free(halphafield);
	free(particles);
	cudaFree(dvfield);
	cudaFree(dffield);
	cudaFree(dalphafield);
	cudaFree(vxfield);
	cudaFree(vyfield);
	cufftDestroy(planr2c);
	cufftDestroy(planc2r);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &vbo);

	sdkDeleteTimer(&timer);
}

int initGL(int* argc, char** argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(wWidth, wHeight);
	glutCreateWindow("Compute Stable Fluids");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(click);
	glutMotionFunc(motion);
	glutReshapeFunc(reshape);


	if (!isGLVersionSupported(1, 5))
	{
		fprintf(stderr, "ERROR: Support for OpenGL 1.5 is missing");
		fflush(stderr);
		return false;
	}

	if (!areGLExtensionsSupported(
		"GL_ARB_vertex_buffer_object"
	))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	return true;
}


int main(int argc, char** argv)
{
	int devID;
	cudaDeviceProp deviceProps;

	printf("%s Starting...\n\n", sSDKname);

	printf("NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");

	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	if (false == initGL(&argc, argv))
	{
		exit(EXIT_SUCCESS);
	}

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	devID = findCudaDevice(argc, (const char**)argv);

	// get number of SMs on this GPU
	checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
	printf("CUDA device [%s] has %d Multi-Processors\n",
		deviceProps.name, deviceProps.multiProcessorCount);

	// Allocate and initialize host data
	GLint bsize;

	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);

	// alpha
	halphafield = (float*)malloc(sizeof(float) * SHORE);
	memset(halphafield, 0, sizeof(float)* SHORE);
	for (int i = 0; i < SHORE; i++)
		halphafield[i] = 2*3.14*i/SHORE;
	cudaMallocPitch((void**)&dalphafield, &tPitch, sizeof(float) * SHORE, 1);
	cudaMemcpy(dalphafield, halphafield, sizeof(float) * SHORE,
		cudaMemcpyHostToDevice);

	// velocity
	hvfield = (cData*)malloc(sizeof(cData) * SHORE);
	memset(hvfield, 0, sizeof(cData) * SHORE);
	for (int i = 0; i < SHORE; i++)
	{
		/*hvfield[i].x = cos(halphafield[i]) * MAX_SPEED;
		hvfield[i].y = sin(halphafield[i]) * MAX_SPEED;*/
		hvfield[i].x = -(myrand()-0.5) / 100;
		hvfield[i].y = -(myrand()-0.5) / 100;
	}
	cudaMallocPitch((void**)&dvfield, &tPitch, sizeof(cData) * SHORE, 1);
	cudaMemcpy(dvfield, hvfield, sizeof(cData) * SHORE,
		cudaMemcpyHostToDevice);

	///force
	hffield = (cData*)malloc(sizeof(cData) * SHORE);
	memset(hffield, 0, sizeof(cData) * SHORE);
	for (int i = 0; i < SHORE; i++)
	{
		hffield[i].y = 0;
		hffield[i].x = 0;
	}
	cudaMallocPitch((void**)&dffield, &tPitch, sizeof(cData) * SHORE, 1);
	cudaMemcpy(dffield, hffield, sizeof(cData) * SHORE,
		cudaMemcpyHostToDevice);

	// localization
	particles = (cData*)malloc(sizeof(cData) * SHORE_ARR);
	memset(particles, 0, sizeof(cData) * SHORE_ARR);

	initParticles(particles, SHORE);

	// Create CUFFT transform plan configuration
	checkCudaErrors(cufftPlan2d(&planr2c, SHORE_ARR, 1, CUFFT_R2C));
	checkCudaErrors(cufftPlan2d(&planc2r, SHORE_ARR, 1, CUFFT_C2R));

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cData) * SHORE_ARR,
		particles, GL_DYNAMIC_DRAW_ARB);

	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize);

	if (bsize != (sizeof(cData) * SHORE_ARR))
		goto EXTERR;

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone));
	getLastCudaError("cudaGraphicsGLRegisterBuffer failed");

	glutCloseFunc(cleanup);
	glutMainLoop();

	return 0;

EXTERR:
	printf("Failed to initialize GL extensions.\n");

	exit(EXIT_FAILURE);
	}
