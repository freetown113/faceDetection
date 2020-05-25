#pragma once
#include <cuda_runtime.h>
#include <npp.h>
#include <cuda.h>
#include <nppi.h>
#include <nppi_geometry_transforms.h>
#include <nppi_data_exchange_and_initialization.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <memory>
#include <cassert>
#include "gridAnchorPlugin.h"
#include "nmsPlugin.h"


#define CUDA_SAFE_CALL( call )											\
{                                										\
    cudaError err = call;                                               \
    if( cudaSuccess != err ) {                                          \
        std::cerr << "Cuda error in file "  << __FILE__ << " in line  " \
	<< __LINE__ << " : " <<  cudaGetErrorString( err ) << std::endl;    \
        cudaDeviceReset();                                              \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

template<typename NvObjectT>
struct NvInferDestructor
{
	void operator()(NvObjectT *obj)
	{
		obj->destroy();
	}
};

template<typename NvObjectT>
using NvInferPtr = std::unique_ptr<NvObjectT, NvInferDestructor<NvObjectT>>;