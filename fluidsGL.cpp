 // OpenGL Graphics includes
#include <helper_gl.h>

#include <GL/freeglut.h>

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

const char* sSDKname = "FishShore";

void cleanup(void);
void reshape(int x, int y);

//Window size data
static int wWidth = 1024;
static int wHeight = 1024;

// Particle data
GLuint vbo = 0;                 // OpenGL vertex buffer object
struct cudaGraphicsResource* cuda_vbo_resource; // handles OpenGL-CUDA exchange
static cData* particles = NULL; // particle positions in host memory

cData* hvfield = NULL;
cData* dvfield = NULL;

int* grid_ids;
int* ids;
int* grid_begin;
int* grid_end;

//Camera movement data
int ox, oy;
int buttonState = 0;
float camera_trans[] = { 0, 0, 1 };
float camera_trans_lag[] = { 0, 0, 1 };
const float inertia = 0.1f;

// Texture pitch
size_t tPitch = 0; // Now this is compatible with gcc in 64-bit


extern "C" void advectParticles(GLuint vbo, cData * v, int* ids, int* grid_ids, int* grid_begin, int* grid_end, int dx, float dt);
extern "C" void test(GLuint vbo, cData * v, int dx, float dt);


void simulateFluids(void)
{
	advectParticles(vbo, dvfield, ids, grid_ids, grid_begin, grid_end, SHORE, DT);
}

void display(void)
{
	simulateFluids();

	// render points from vertex buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	for (int c = 0; c < 3; ++c)
	{
		camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
	}

	glPushMatrix();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(90, 1, 0.1, 8);

	glMatrixMode(GL_MODELVIEW);


	float r = 1.5f * camera_trans_lag[2];
	float r_horizontal = r * cos(camera_trans_lag[0]);

	gluLookAt(
		0.5 + r_horizontal * cos(camera_trans_lag[1]),
		0.5 + r_horizontal * sin(camera_trans_lag[1]),
		0.5 + r * sin(camera_trans_lag[0]),
		0.5, 0.5, 0.5, 0, 1, 0);
	

	glColor4f(1, 0, 0, 0.5f);
	glPointSize(1);
	glBegin(GL_LINES);
	glVertex3f(MIN_DIM, MIN_DIM, MIN_DIM);
	glVertex3f(MAX_DIM, MIN_DIM, MIN_DIM);
	glVertex3f(MIN_DIM, MIN_DIM, MIN_DIM);
	glVertex3f(MIN_DIM, MAX_DIM, MIN_DIM);
	glVertex3f(MIN_DIM, MIN_DIM, MIN_DIM);
	glVertex3f(MIN_DIM, MIN_DIM, MAX_DIM);
	glVertex3f(MAX_DIM, MAX_DIM, MAX_DIM);
	glVertex3f(MIN_DIM, MAX_DIM, MAX_DIM);
	glVertex3f(MAX_DIM, MAX_DIM, MAX_DIM);
	glVertex3f(MAX_DIM, MIN_DIM, MAX_DIM);
	glVertex3f(MAX_DIM, MAX_DIM, MAX_DIM);
	glVertex3f(MAX_DIM, MAX_DIM, MIN_DIM);
	glVertex3f(MAX_DIM, MIN_DIM, MIN_DIM);
	glVertex3f(MAX_DIM, MAX_DIM, MIN_DIM);
	glVertex3f(MAX_DIM, MIN_DIM, MIN_DIM);
	glVertex3f(MAX_DIM, MIN_DIM, MAX_DIM);
	glVertex3f(MIN_DIM, MAX_DIM, MIN_DIM);
	glVertex3f(MAX_DIM, MAX_DIM, MIN_DIM);
	glVertex3f(MIN_DIM, MAX_DIM, MIN_DIM);
	glVertex3f(MIN_DIM, MAX_DIM, MAX_DIM);
	glVertex3f(MIN_DIM, MIN_DIM, MAX_DIM);
	glVertex3f(MAX_DIM, MIN_DIM, MAX_DIM);
	glVertex3f(MIN_DIM, MIN_DIM, MAX_DIM);
	glVertex3f(MIN_DIM, MAX_DIM, MAX_DIM);

	
	glEnd();


	glColor4f(0, 1, 0, 0.5f);
	glPointSize(1);

	glEnableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(3, GL_FLOAT, 0, NULL);
	glDrawArrays(GL_LINES, 0, SHORE_ARR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);


	glPopMatrix();



	glutSwapBuffers();


	//char fps[256];
	//sprintf(fps, "Cuda/GL Fish Shore (%d x %d)", SHORE, 1);
	//glutSetWindowTitle(fps);


	glutPostRedisplay();
}

float myrand(void)
{
	return rand() / (float)RAND_MAX;
}

void initParticles(cData* p, int shore_count)
{
	for (int i = 0; i < shore_count; i++)
	{
		p[2 * i].x = (myrand() / 2 + MIN_DIM);
		p[2 * i].y = (myrand() / 2 + MIN_DIM);
#ifdef Z_AXIS
		p[2 * i].z = (myrand() / 2 + MIN_DIM);
#else
		p[2 * i].z = 0;
#endif // Z_AXIS

		p[2 * i + 1].x = p[2 * i].x + 0.02f;
		p[2 * i + 1].y = p[2 * i].y + 0.02f;
		p[2 * i + 1].z = p[2 * i].z;
	}
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:
		glutDestroyWindow(glutGetWindow());
		return;

	default:
		break;
	}
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - ox);
	dy = (float)(y - oy);

	if (buttonState == 3)
	{
		// left+middle = zoom
		camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
	}
	else if (buttonState & 2)
	{
		// middle = translate
		
	}
	else if (buttonState & 1)
	{
		// left = rotate
		camera_trans[0] += dx / 100.0f;
		camera_trans[1] -= dy / 100.0f;
	}


	ox = x;
	oy = y;

	glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
	int mods;

	if (state == GLUT_DOWN)
	{
		buttonState |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		buttonState = 0;
	}

	mods = glutGetModifiers();

	if (mods & GLUT_ACTIVE_SHIFT)
	{
		buttonState = 2;
	}
	else if (mods & GLUT_ACTIVE_CTRL)
	{
		buttonState = 3;
	}

	ox = x;
	oy = y;

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
	free(particles);
	cudaFree(dvfield);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &vbo);
}

int initGL(int* argc, char** argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(wWidth, wHeight);
	glutCreateWindow("Compute Stable Fluids");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
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

	// velocity
	hvfield = (cData*)malloc(sizeof(cData) * SHORE);
	memset(hvfield, 0, sizeof(cData) * SHORE);
	for (int i = 0; i < SHORE; i++)
	{
		hvfield[i].x = -(myrand() - 0.5f) / 100;
		hvfield[i].y = -(myrand() - 0.5f) / 100;
		hvfield[i].z = -(myrand() - 0.5f) / 100;
	}
	cudaMallocPitch((void**)&dvfield, &tPitch, sizeof(cData) * SHORE, 1);
	cudaMemcpy(dvfield, hvfield, sizeof(cData) * SHORE,
		cudaMemcpyHostToDevice);

	// localization
	particles = (cData*)malloc(sizeof(cData) * SHORE_ARR);
	memset(particles, 0, sizeof(cData) * SHORE_ARR);

	initParticles(particles, SHORE);

	//grid arrays
	checkCudaErrors(cudaMalloc((void**)&ids, sizeof(int) * SHORE));
	checkCudaErrors(cudaMalloc((void**)&grid_ids, sizeof(int) * SHORE));
	checkCudaErrors(cudaMalloc((void**)&grid_begin, sizeof(int) * (GRID_SIZE* GRID_SIZE* GRID_SIZE)));
	checkCudaErrors(cudaMalloc((void**)&grid_end, sizeof(int) * (GRID_SIZE* GRID_SIZE* GRID_SIZE)));



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

	if (!TEST_TIME)
	{
		glutCloseFunc(cleanup);
		glutMainLoop();
	}
	else
	{
		test(vbo, dvfield, SHORE, DT);
		cleanup();
	}

	return 0;

EXTERR:
	printf("Failed to initialize GL extensions.\n");

	exit(EXIT_FAILURE);
}
