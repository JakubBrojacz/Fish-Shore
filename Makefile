OS_SUFFIX = linux/aarch64

CU_SRCS += \
src/fish_kernels.cu 

CPP_SRCS += \
src/fish.cpp 

OBJS += \
obj/fish.o \
obj/fish_kernels.o 

CU_DEPS += \
obj/fish_kernels.d 

CPP_DEPS += \
obj/fish.d 


USER_OBJS := /opt/cuda/samples/common/lib/$(OS_SUFFIX)/libGLEW.a

LIBS := -lcufft



GLUT_LIBS := -lGL -lGLU -lglut 


all: fish

fish: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC Linker'
	/opt/cuda/bin/nvcc --cudart static -L"/opt/cuda/samples/common/lib/linux/x86_64" $(GLUT_LIBS) --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61 -link -o  "fish" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '



obj/%.o: src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/bin/nvcc -I"/opt/cuda/samples/5_Simulations" -I"/opt/cuda/samples/common/inc" -I"scr" -G -g -O0 -gencode arch=compute_61,code=sm_61  -odir "obj" -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/bin/nvcc -I"/opt/cuda/samples/5_Simulations" -I"/opt/cuda/samples/common/inc" -I"scr" -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

obj/%.o: src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/bin/nvcc -I"/opt/cuda/samples/5_Simulations" -I"/opt/cuda/samples/common/inc" -I"scr" -G -g -O0 -gencode arch=compute_61,code=sm_61  -odir "obj" -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/bin/nvcc -I"/opt/cuda/samples/5_Simulations" -I"/opt/cuda/samples/common/inc" -I"scr" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

