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
#include "fish_kernels.h"

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

int count = 0;

// Particle data (second type)
GLuint vbo_1 = 0;                 // OpenGL vertex buffer object
struct cudaGraphicsResource* cuda_vbo_resource_1; // handles OpenGL-CUDA exchange
static cData* particles_1 = NULL; // particle positions in host memory

cData* hvfield_1 = NULL;
cData* dvfield_1 = NULL;

int count_1 = 0;


// Grid data
int* grid_ids;
int* ids;
int* grid_begin;
int* grid_end;


//Camera movement data
int ox, oy;
int buttonState = 0;
float camera_trans[] = { 0, 0, 0.5 };
float camera_trans_lag[] = { 0, 0, 0.5 };
const float inertia = 0.1f;

// Texture pitch
size_t tPitch = 0; // Now this is compatible with gcc in 64-bit




extern "C" void advectParticles(GLuint vbo, cData * v, struct cudaGraphicsResource* cuda_vbo_resource, int* ids, int* grid_ids, int* grid_begin, int* grid_end, int dx, float dt);
extern "C" void test(GLuint vbo, cData * v, int* ids, int* grid_ids, int* grid_begin, int* grid_end, int dx, float dt);


void simulateFluids(void)
{
	advectParticles(vbo, dvfield, cuda_vbo_resource, ids, grid_ids, grid_begin, grid_end, count, DT);
	advectParticles(vbo_1, dvfield_1, cuda_vbo_resource_1, ids, grid_ids, grid_begin, grid_end, count_1, DT);
}

void display(void)
{
	simulateFluids();

	// render points from vertex buffer
	glClearColor(1, 1, 1, 1);
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


	glPointSize(1);

	glColor4f(0,0,0, 128.f / 256);
	glEnableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(3, GL_FLOAT, 0, NULL);
	glDrawArrays(GL_LINES, 0, SHORE_ARR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glColor4f(2.f / 256, 41.f / 256, 147.f / 256, 128.f / 256);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_1);
	glVertexPointer(3, GL_FLOAT, 0, NULL);
	glDrawArrays(GL_LINES, 0, SHORE_ARR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);


	glPopMatrix();



	glutSwapBuffers();

	glutPostRedisplay();
}


struct Bitmap
{
	unsigned char* data;
	int width;
	int height;

	Bitmap(unsigned char* data, int width, int height) : data(data), width(width), height(height) {};
};
Bitmap* readBMP(char* filename)
{
	int i;
	FILE* f = fopen(filename, "rb");
	unsigned char info[54];
	fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

	// extract image height and width from header
	int width = *(int*)&info[18];
	int height = *(int*)&info[22];

	int size = 3 * width * height;
	unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
	fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
	fclose(f);

	for (i = 0; i < size; i += 3)
	{
		unsigned char tmp = data[i];
		data[i] = data[i + 2];
		data[i + 2] = tmp;
	}

	return new Bitmap(data, width, height);
}

float myrand(void)
{
	return rand() / (float)RAND_MAX;
}

void initParticles(cData* p, cData* p_1, int shore_count)
{
	Bitmap* bmp = readBMP("./resources/logo_mini.bmp");

	int iter = 0;
	int gap = 8;
	printf("%i, %i, %i\n", bmp->data[0], bmp->data[1], bmp->data[2]);
	for (int i = 0; i < bmp->width; i++)
		for (int j = 0; j < bmp->height; j++)
			if (iter < shore_count*gap && bmp->data[3 * (i + j * (bmp->width))]<125)
			{
				if (iter % gap == 0 && bmp->data[3 * (i + j * (bmp->width)) + 2] < 125)
				{
					//printf("%i, %i, %i\n", bmp->data[0], bmp->data[1], bmp->data[2], bmp->data[3 * (i + j * (bmp->width))]);
					int id = count;
					p[2 * id].z = (MIN_DIM + (MAX_DIM - MIN_DIM) * ((float)i / bmp->width));
					p[2 * id].y = (MIN_DIM + (MAX_DIM - MIN_DIM) * ((float)j / bmp->height));
					p[2 * id].x = ((myrand()-0.5f) / 20 + (MIN_DIM+MAX_DIM)/2);

					p[2 * id + 1].x = p[2 * id].x + 0.001;
					p[2 * id + 1].y = p[2 * id].y + 0.001;
					p[2 * id + 1].z = p[2 * id].z;

					count++;
				}
				if (iter % gap == 0 && bmp->data[3 * (i + j * (bmp->width)) + 2] > 125)
				{
					//printf("%i, %i, %i\n", bmp->data[0], bmp->data[1], bmp->data[2], bmp->data[3 * (i + j * (bmp->width))]);
					int id = count_1;
					p_1[2 * id].z = (MIN_DIM + (MAX_DIM - MIN_DIM) * ((float)i / bmp->width));
					p_1[2 * id].y = (MIN_DIM + (MAX_DIM - MIN_DIM) * ((float)j / bmp->height));
					p_1[2 * id].x = ((myrand() - 0.5f) / 20 + (MIN_DIM + MAX_DIM) / 2);

					p_1[2 * id + 1].x = p_1[2 * id].x + 0.001;
					p_1[2 * id + 1].y = p_1[2 * id].y + 0.001;
					p_1[2 * id + 1].z = p_1[2 * id].z;

					count_1++;
				}
				
				iter++;
			}
	printf("count: %i\n", count);
	printf("count 1: %i\n", count_1);
//	for (int i = count; i < shore_count; i++)
//	{
//		p[2 * i].x = (myrand() / 2 + MIN_DIM);
//		p[2 * i].y = (myrand() / 2 + MIN_DIM);
//#ifdef Z_AXIS
//		p[2 * i].z = (myrand() / 2 + MIN_DIM);
//#else
//		p[2 * i].z = 0;
//#endif // Z_AXIS
//
//		p[2 * i + 1].x = p[2 * i].x + 0.02f;
//		p[2 * i + 1].y = p[2 * i].y + 0.02f;
//		p[2 * i + 1].z = p[2 * i].z;
//	}

	delete bmp;
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

	free(hvfield_1);
	free(particles_1);
	cudaFree(dvfield_1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &vbo);
	glDeleteBuffers(1, &vbo_1);
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

	hvfield_1 = (cData*)malloc(sizeof(cData) * SHORE);
	memset(hvfield_1, 0, sizeof(cData) * SHORE);
	for (int i = 0; i < SHORE; i++)
	{
		hvfield_1[i].x = -(myrand() - 0.5f) / 100;
		hvfield_1[i].y = -(myrand() - 0.5f) / 100;
		hvfield_1[i].z = -(myrand() - 0.5f) / 100;
	}
	cudaMallocPitch((void**)&dvfield_1, &tPitch, sizeof(cData) * SHORE, 1);
	cudaMemcpy(dvfield_1, hvfield_1, sizeof(cData) * SHORE,
		cudaMemcpyHostToDevice);

	// localization
	particles = (cData*)malloc(sizeof(cData) * SHORE_ARR);
	memset(particles, 0, sizeof(cData) * SHORE_ARR);

	particles_1 = (cData*)malloc(sizeof(cData) * SHORE_ARR);
	memset(particles_1, 0, sizeof(cData) * SHORE_ARR);

	initParticles(particles, particles_1, SHORE);

	//grid arrays
	checkCudaErrors(cudaMalloc((void**)&ids, sizeof(int) * SHORE));
	checkCudaErrors(cudaMalloc((void**)&grid_ids, sizeof(int) * SHORE));
	checkCudaErrors(cudaMalloc((void**)&grid_begin, sizeof(int) * (GRID_SIZE * GRID_SIZE * GRID_SIZE)));
	checkCudaErrors(cudaMalloc((void**)&grid_end, sizeof(int) * (GRID_SIZE * GRID_SIZE * GRID_SIZE)));



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


	glGenBuffers(1, &vbo_1);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_1);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cData) * SHORE_ARR,
		particles_1, GL_DYNAMIC_DRAW_ARB);

	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize);

	if (bsize != (sizeof(cData) * SHORE_ARR))
		goto EXTERR;

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource_1, vbo_1, cudaGraphicsMapFlagsNone));
	getLastCudaError("cudaGraphicsGLRegisterBuffer failed");

	if (!TEST_TIME)
	{
		glutCloseFunc(cleanup);
		glutMainLoop();
	}
	else
	{
		test(vbo, dvfield, ids, grid_ids, grid_begin, grid_end, SHORE, DT);
		cleanup();
	}

	return 0;

EXTERR:
	printf("Failed to initialize GL extensions.\n");

	exit(EXIT_FAILURE);
}
