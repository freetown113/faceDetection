# Result name for lib
#NAME_L	= libtensorrt.so
NAME_UTILS = libbboxUtils.so
NAME_COMMON = libcommon.so
NAME_MATH = libreducedMath.so
NAME_GRID = libgridAnchorInference.so
NAME_CLASS = libsortScoresPerClass.so
NAME_IMAGE = libsortScoresPerImage.so
NAME_PERMUTE = libpermuteData.so
NAME_DECODE = libdecodeBBoxes.so
NAME_GATHER = libgatherTopDetections.so
NAME_NMS = liballClassNMS.so
NAME_DETECT = libdetectionInference.so
NAME_NMSPLUG = libnmsPlugin.so
NAME_DECODE_PP = libdecodeBBoxesPP.so
NAME_GATHER_PP = libgatherTopDetectionsPP.so
NAME_DETECT_PP = libdetectionInferencePP.so

################################################################################
# Directory layout

SRC_DIR 	= ./src
INCLUDE_DIR	= -I../include -I/home/asorokin/projects/cub/cub \
				#-I/capital/3030/SorokinA/prj_n/onnx2trt/TensorRT-7.0.0.11/include
BUILD_DIR	= ./build
LIB_DIR		= ./lib
RESULT_UTILS		= $(LIB_DIR)/$(NAME_UTILS)
RESULT_COMMON		= $(LIB_DIR)/$(NAME_COMMON)
RESULT_MATH			= $(LIB_DIR)/$(NAME_MATH)
RESULT_GRID			= $(LIB_DIR)/$(NAME_GRID)
RESULT_CLASS		= $(LIB_DIR)/$(NAME_CLASS)
RESULT_IMAGE		= $(LIB_DIR)/$(NAME_IMAGE)
RESULT_PERMUTE		= $(LIB_DIR)/$(NAME_PERMUTE)
RESULT_DECODE		= $(LIB_DIR)/$(NAME_DECODE)
RESULT_GATHER		= $(LIB_DIR)/$(NAME_GATHER)
RESULT_NMS			= $(LIB_DIR)/$(NAME_NMS)
RESULT_DETECT		= $(LIB_DIR)/$(NAME_DETECT)
RESULT_NMSPLUG		= $(LIB_DIR)/$(NAME_NMSPLUG)
RESULT_DECODE_PP	= $(LIB_DIR)/$(NAME_DECODE_PP)
RESULT_GATHER_PP	= $(LIB_DIR)/$(NAME_GATHER_PP)
RESULT_DETECT_PP	= $(LIB_DIR)/$(NAME_DETECT_PP)


OPTIMAZE_COMMON	= -O2 -std=c++11 -g

# Processor specific optimization (safe for debugging)
OPTIMAZE_SPECIFIC = -fPIC -std=c++11

# Optimization options
OPTIMAZE	= $(OPTIMAZE_COMMON) $(OPTIMAZE_SPECIFIC)

# Compiler
CC		= gcc
DEVCC		= /usr/local/cuda-10.2/bin/nvcc
# Linker
LINK		= /usr/local/cuda-10.2/bin/nvcc

#gencode
GEN		= 75
ARCH		= -gencode arch=compute_$(GEN),code=sm_$(GEN)

# Compiler flags
GENFLAGS    = -c $(ARCH) $(ENV_FLAGS) -std=c++11
CFLAGS        = $(GENFLAGS) -Xcompiler "$(OPTIMAZE)" 

# Linker flags
#LIB             = $(ARCH) -ccbin=$(CC) -Xcompiler "$(OPTIMAZE_SPECIFIC)" -L/opt/TensorRT-7.0.0.11/lib -lmyelin -lnvonnxparser -lnvinfer -lnvparsers -lnvinfer_plugin -L/usr/local/cuda-10.2/lib64 -lnppial -lnppidei -lnppig
LIB             = $(ARCH) -ccbin=$(CC) -Xcompiler "$(OPTIMAZE_SPECIFIC)" -L/opt/TensorRT-7.0.0.11/lib -lmyelin -lnvonnxparser -lnvinfer -lnvparsers -lnvinfer_plugin -L/usr/local/cuda-10.2/lib64 -lnppial -lnppidei -lnppig -L./lib -lallClassNMS -ldecodeBBoxes -lpermuteData -lsortScoresPerImage -lsortScoresPerImage -lgatherTopDetections -ldecodeBBoxesPP -lgatherTopDetectionsPP

OBJS_UTILS = \
        $(BUILD_DIR)/bboxUtils.o \

OBJS_COMMON = \
		$(BUILD_DIR)/common.o \

OBJS_MATH = \
       $(BUILD_DIR)/reducedMath.o \

OBJS_GRID = \
       $(BUILD_DIR)/gridAnchorInference.o \

OBJS_CLASS = \
       $(BUILD_DIR)/sortScoresPerClass.o \

OBJS_IMAGE = \
       $(BUILD_DIR)/sortScoresPerImage.o \

OBJS_PERMUTE = \
       $(BUILD_DIR)/permuteData.o \

OBJS_DECODE = \
       $(BUILD_DIR)/decodeBBoxes.o \

OBJS_GATHER = \
       $(BUILD_DIR)/gatherTopDetections.o \

OBJS_NMS = \
       $(BUILD_DIR)/allClassNMS.o \

OBJS_DETECT = \
		$(BUILD_DIR)/detectionInference.o \

OBJS_NMSPLUG = \
		$(BUILD_DIR)/nmsPlugin.o \

OBJS_DECODE_PP = \
       $(BUILD_DIR)/decodeBBoxesPP.o \

OBJS_GATHER_PP = \
       $(BUILD_DIR)/gatherTopDetectionsPP.o \

OBJS_DETECT_PP = \
		$(BUILD_DIR)/detectionInferencePP.o \


lib_utils : $(RESULT_UTILS)
$(RESULT_UTILS) : $(OBJS_UTILS)
	$(LINK) -shared $(LIB) $(OBJS_UTILS) -o $(RESULT_UTILS)

lib_common : $(RESULT_COMMON)
$(RESULT_COMMON) : $(OBJS_COMMON)
	$(LINK) -shared $(LIB) $(OBJS_COMMON) -o $(RESULT_COMMON)

lib_math : $(RESULT_MATH)
$(RESULT_MATH) : $(OBJS_MATH)
	$(LINK) -shared $(LIB) $(OBJS_MATH) -o $(RESULT_MATH)

lib_grid : $(RESULT_GRID)
$(RESULT_GRID) : $(OBJS_GRID)
	$(LINK) -shared $(LIB) $(OBJS_GRID) -o $(RESULT_GRID)

lib_class : $(RESULT_CLASS)
$(RESULT_CLASS) : $(OBJS_CLASS)
	$(LINK) -shared $(LIB) $(OBJS_CLASS) -o $(RESULT_CLASS)

lib_image : $(RESULT_IMAGE)
$(RESULT_IMAGE) : $(OBJS_IMAGE)
	$(LINK) -shared $(LIB) $(OBJS_IMAGE) -o $(RESULT_IMAGE)

lib_permute : $(RESULT_PERMUTE)
$(RESULT_PERMUTE) : $(OBJS_PERMUTE)
	$(LINK) -shared $(LIB) $(OBJS_PERMUTE) -o $(RESULT_PERMUTE)

lib_decode : $(RESULT_DECODE)
$(RESULT_DECODE) : $(OBJS_DECODE)
	$(LINK) -shared $(LIB) $(OBJS_DECODE) -o $(RESULT_DECODE)

lib_gather : $(RESULT_GATHER)
$(RESULT_GATHER) : $(OBJS_GATHER)
	$(LINK) -shared $(LIB) $(OBJS_GATHER) -o $(RESULT_GATHER)

lib_nms : $(RESULT_NMS)
$(RESULT_NMS) : $(OBJS_NMS)
	$(LINK) -shared $(LIB) $(OBJS_NMS) -o $(RESULT_NMS)

lib_detect : $(RESULT_DETECT)
$(RESULT_DETECT) : $(OBJS_DETECT)
	$(LINK) -shared $(LIB) $(OBJS_DETECT) -o $(RESULT_DETECT)

lib_nmsplug : $(RESULT_NMSPLUG)
$(RESULT_NMSPLUG) : $(OBJS_NMSPLUG)
	$(LINK) -shared $(LIB) $(OBJS_NMSPLUG) -o $(RESULT_NMSPLUG)

lib_decode_PP : $(RESULT_DECODE_PP)
$(RESULT_DECODE_PP) : $(OBJS_DECODE_PP)
	$(LINK) -shared $(LIB) $(OBJS_DECODE_PP) -o $(RESULT_DECODE_PP)

lib_gather_PP : $(RESULT_GATHER_PP)
$(RESULT_GATHER_PP) : $(OBJS_GATHER_PP)
	$(LINK) -shared $(LIB) $(OBJS_GATHER_PP) -o $(RESULT_GATHER_PP)

lib_detect_PP : $(RESULT_DETECT_PP)
$(RESULT_DETECT_PP) : $(OBJS_DETECT_PP)
	$(LINK) -shared $(LIB) $(OBJS_DETECT_PP) -o $(RESULT_DETECT_PP)

main_dsod : main_dsod.cpp
	$(DEVCC) $(INCLUDE_DIR) -ccbin=$(CC) $(OPTIMAZE_COMMON) `pkg-config --libs opencv` -L./lib -ldsod $< -o main.exe
 
main_dsod2 : main_dsod2.cpp
	$(DEVCC) $(INCLUDE_DIR) -ccbin=$(CC) $(OPTIMAZE_COMMON) `pkg-config --libs opencv` -L./lib -ldsod2 $< -o main.exe
 
main_dsod3 : main_dsod3.cpp
	$(DEVCC) $(INCLUDE_DIR) -ccbin=$(CC) $(OPTIMAZE_COMMON) `pkg-config --libs opencv` -L./lib -ldsod3 $< -o main.exe

main_dsod4 : main_dsod4.cpp
	$(DEVCC) $(INCLUDE_DIR) -ccbin=$(CC) $(OPTIMAZE_COMMON) `pkg-config --libs opencv` -L./lib -ldsod4 $< -o main.exe

main_tface : main_tface.cpp
	$(DEVCC) $(INCLUDE_DIR) -ccbin=$(CC) $(OPTIMAZE_COMMON) `pkg-config --libs opencv` -L./lib -ltensorrt $< -o main.exe
	
main_resnet : main_resnet.cpp
	$(DEVCC) $(INCLUDE_DIR) -ccbin=$(CC) $(OPTIMAZE_COMMON) `pkg-config --libs opencv` -L./lib -ltensorrt $< -o main.exe

main_sfd : main_sfd.cpp
	$(DEVCC) $(INCLUDE_DIR) -ccbin=$(CC) $(OPTIMAZE_COMMON) `pkg-config --libs opencv` -L./lib -ltensorrt $< -o main.exe

main_shuf : main_shuf.cpp
	$(DEVCC) $(INCLUDE_DIR) -ccbin=$(CC) $(OPTIMAZE_COMMON) `pkg-config --libs opencv` -L./lib -lshuf $< -o main.exe

clean : clean_l

clean_l :
	rm -f $(OBJS_UTILS) $(OBJS_COMMON) $(OBJS_MATH) $(OBJS_GRID) $(OBJS_CLASS) $(OBJS_IMAGE) $(OBJS_PERMUTE) $(OBJS_DECODE) $(OBJS_GATHER) $(RESULT_NMS) $(RESULT_DETECT) $(RESULT_NMSPLUG) $(OBJS_DETECT_PP) $(OBJS_NMS_PP) $(OBJS_GATHER_PP) $(OBJS_DECODE_PP)

# Individual files build rules
$(BUILD_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(DEVCC) $(INCLUDE_DIR) -ccbin=$(CC) $(CFLAGS) $< -o $@

$(BUILD_DIR)/%.o : $(SRC_DIR)/%.cu
	$(DEVCC) $(INCLUDE_DIR) -ccbin=$(CC) $(CFLAGS) $< -o $@
