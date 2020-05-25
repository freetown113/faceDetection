#include "kernel.h"
#include "reducedMath.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "plugin.h"
#include<NvInferPlugin.h>
#include<NvInferPluginUtils.h>
#include<NvInfer.h>
#include<NvInferRuntime.h>


template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
__global__
void fill_i_j(int n, float *x, float *y)
{
	int dim = n*n;
	//int i = blockIdx.x*blockDim.x*blockDim.x + threadIdx.x;
	//if (i < n*n)
	for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x;
		i < dim; i += gridDim.x * nthdsPerCTA)
	{
		//printf("i_j %d\n", i);
		x[i] = i / n;
		y[i] = i - (i / n)*n;
		//printf("i_j %d\n", i - (i / n)*n);
	}
}


template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
__global__
void fill_mean(int n, float *x, float *y, float *x_old, float *y_old,
	float *sf, float *cf, float f_k, float s_k, float prime)
{
	//printf("s_k %f, f_k %f\n", s_k, f_k);
	int dim = n*n;
	float indent = 0.5;
	//int i = blockIdx.x*blockDim.x*blockDim.x + threadIdx.x;
	//if (i < n*n*2)
	for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x;
		i < dim; i += gridDim.x * nthdsPerCTA)
	{
		//printf("mean %d\n", i);
		x[i] = __saturatef((y_old[i] + indent) / f_k);
		y[i] = __saturatef((x_old[i] + indent) / f_k);
		/*sf[i*(i % 2)] = __saturatef(prime);
		cf[i*(i % 2)] = __saturatef(prime);
		sf[i - i * (i % 2)] = __saturatef(s_k);
		cf[i - i * (i % 2)] = __saturatef(s_k);*/
		sf[i] = __saturatef(s_k);
		cf[i] = __saturatef(s_k);
	}
}


template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
__global__
void fill_final(int n, float *x, float *y, float *sf, float *cf, float* d_fin)
{
	int dim = n*n;
	float* d_var = d_fin + n * n * 4;
	//int i = blockIdx.x*blockDim.x*blockDim.x + threadIdx.x;
	//if (i < n*n * 2)
	for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x;
		i < dim; i += gridDim.x * nthdsPerCTA)
	{
		d_fin[i * 4] = x[i];
		d_fin[i * 4 + 1] = y[i];
		d_fin[i * 4 + 2] = sf[i];
		d_fin[i * 4 + 3] = cf[i];
		//d_fin[i * 4] = __saturatef(x[i] - sf[i]/2);
		//d_fin[i * 4 + 1] = __saturatef(y[i] - cf[i]/2);
		//d_fin[i * 4 + 2] = __saturatef(x[i] + sf[i]/2);
		//d_fin[i * 4 + 3] = __saturatef(y[i] + cf[i]/2);
		d_var[i * 4] = 0.1;
		d_var[i * 4 + 1] = 0.1;
		d_var[i * 4 + 2] = 0.2;
		d_var[i * 4 + 3] = 0.2;
	}
}









//using nvinfer1::rt::reduced_divisor;
//template <unsigned nthdsPerCTA>
//__launch_bounds__(nthdsPerCTA)
//__global__ void gridAnchorKernel(
//	const nvinfer1::plugin::GridAnchorParameters param,
//	const int numAspectRatios,
//	reduced_divisor divObj,
//	const float* widths,
//	const float* heights,
//	float* outputData
//)
//{
//	// output dims: (H, W, param.numMinSize, (1+haveMaxSize+numAR-1), 4)
//	const int dim = param.H * param.W * numAspectRatios;
//	/*
//	 * Parameters used to calculate the bounding box coordinates back to input image scale
//	 * Normally we calculate the anchorStride = image_input_size (in pixel) / feature_map_size
//	 * Here we do not use image_input_size for the moment
//	 * Instead we use 1.0
//	 * The coordinates calculated are scaled by the input image size.
//	 * Most of the coordinates will be in a range of [0, 1], except for the bounding box coordinates going outside of the image
//	 * Every coordinate will go back to the pixel coordinates in the input image if being multiplied by image_input_size
//	 * Here we implicitly assumes the image input and feature map are square
//	 */
//	float anchorStride = (1.0 / param.H);
//	float anchorOffset = 0.5 * anchorStride;
//
//	int tid = blockIdx.x * blockDim.x + threadIdx.x;
//	if (tid >= dim)
//		return;
//	int arId, currIndex;
//	divObj.divmod(tid, currIndex, arId);
//
//	const int w = currIndex % param.W;
//	const int h = currIndex / param.W;
//
//	// Center coordinates
//	float yC = h * anchorStride + anchorOffset;
//	float xC = w * anchorStride + anchorOffset;
//
//	// x_min, y_min
//	float xMin = xC - 0.5 * widths[arId];
//	float yMin = yC - 0.5 * heights[arId];
//
//	// x_max, y_max
//	float xMax = xC + 0.5 * widths[arId];
//	float yMax = yC + 0.5 * heights[arId];
//
//	outputData[tid * 4] = xMin;
//	outputData[tid * 4 + 1] = yMin;
//	outputData[tid * 4 + 2] = xMax;
//	outputData[tid * 4 + 3] = yMax;
//
//	// Remember to move the output cursor
//	float* output = outputData + dim * 4;
//
//	// Simply copying the variance
//	output[tid * 4] = param.variance[0];
//	output[tid * 4 + 1] = param.variance[1];
//	output[tid * 4 + 2] = param.variance[2];
//	output[tid * 4 + 3] = param.variance[3];
//
//}

pluginStatus_t anchorGridInference(
	cudaStream_t stream,
	const nvinfer1::plugin::GridAnchorParameters param,
	const int numAspectRatios,
	const void* widths,
	const void* heights,
	void* outputData
)
{
	float *d_x, *d_y, *d_cx, *d_cy, *d_sf, *d_cf;
	const int dim = param.H * param.W * numAspectRatios;
	if (dim > 5120)
	{
		std::cout << "5120" << std::endl;
		const int BS = 128;
		const int GS = (dim + BS - 1) / BS;
		
		cudaMalloc(&d_x, param.H * param.W * sizeof(float));
		cudaMalloc(&d_y, param.H * param.W * sizeof(float));
		fill_i_j<BS> << < (param.H * param.W + 128) / 128, BS >> > (param.H, d_x, d_y);
		cudaMalloc(&d_cx, param.H * param.W*numAspectRatios * 2 * sizeof(float));
		cudaMalloc(&d_cy, param.H * param.W*numAspectRatios * 2 * sizeof(float));
		cudaMalloc(&d_sf, param.H * param.W*numAspectRatios * 2 * sizeof(float));
		cudaMalloc(&d_cf, param.H * param.W*numAspectRatios * 2 * sizeof(float));
		fill_mean<BS> << < (param.H * param.W + 128) / 128, BS >> > (param.H, d_cx, d_cy, d_x, d_y, d_sf, d_cf,
			512 / 128, 307.2 / 512, std::sqrt(307.2 / 512 * 384.0 / 512));
		
		fill_final<BS> << < (param.H * param.W + 256) / 256, BS >> > (param.H, d_cx, d_cy, d_sf, d_cf, (float*)outputData);
		CSC(cudaGetLastError(), STATUS_FAILURE);
		cudaFree(d_x);
		cudaFree(d_y);
		cudaFree(d_cx);
		cudaFree(d_cy);
		cudaFree(d_sf);
		cudaFree(d_cf);
		return STATUS_SUCCESS;
	}
	else
	{
		std::cout << "OTHER" << std::endl;
		const int BS = 32;
		const int GS = (dim + BS - 1) / BS;
		cudaMalloc(&d_x, param.H * param.W * 2 * sizeof(float));
		cudaMalloc(&d_y, param.H * param.W * 2 * sizeof(float));
		//std::cout << param.H << " " << param.W << std::endl;
		fill_i_j<BS> << < (param.H * param.W + 128) / 128, BS >> > (param.H, d_x, d_y);
		cudaMalloc(&d_cx, param.H * param.W*numAspectRatios * 2 * sizeof(float));
		cudaMalloc(&d_cy, param.H * param.W*numAspectRatios * 2 * sizeof(float));
		cudaMalloc(&d_sf, param.H * param.W*numAspectRatios * 2 * sizeof(float));
		cudaMalloc(&d_cf, param.H * param.W*numAspectRatios * 2 * sizeof(float));
		//fill_mean<BS> << < (param.H * param.W + 128) / 128, BS >> > (param.H, d_cx, d_cy, d_x, d_y, d_sf, d_cf,
		//	512 / 128, 307.2 / 512,	std::sqrt(307.2 / 512 * 384.0 / 512));
		fill_mean<BS> << < (param.H * param.W + 128) / 128, BS >> > (param.H, d_cx, d_cy, d_x, d_y, d_sf, d_cf,
			640.0f / (float)param.minSize, (float)param.maxSize / 640.0f, std::sqrt(1));
		/*fill_mean<BS> << < (param.H * param.W + 128) / 128, BS >> > (param.H, d_cx, d_cy, d_x, d_y, d_sf, d_cf,
			640.0f / 32.0f, 64.0f / 640.0f, std::sqrt(1));*/

		fill_final<BS> << < (param.H * param.W + 128) / 128, BS >> > (param.H, d_cx, d_cy, d_sf, d_cf, (float*)outputData);
		
		CSC(cudaGetLastError(), STATUS_FAILURE);
		cudaFree(d_x);
		cudaFree(d_y);
		cudaFree(d_cx);
		cudaFree(d_cy);
		cudaFree(d_sf);
		cudaFree(d_cf);
		return STATUS_SUCCESS;
	}




	//const int dim = param.H * param.W * numAspectRatios;
	//reduced_divisor divObj(numAspectRatios);
	//if (dim > 5120)
	//{
	//	const int BS = 128;
	//	const int GS = (dim + BS - 1) / BS;
	//	gridAnchorKernel<BS> << <GS, BS, 0, stream >> > (param, numAspectRatios, divObj,
	//		(const float*)widths, (const float*)heights,
	//		(float*)outputData);

	//}
	//else
	//{
	//	const int BS = 32;
	//	const int GS = (dim + BS - 1) / BS;
	//	gridAnchorKernel<BS> << <GS, BS, 0, stream >> > (param, numAspectRatios, divObj,
	//		(const float*)widths, (const float*)heights,
	//		(float*)outputData);

	//}
	//CSC(cudaGetLastError(), STATUS_FAILURE);
	//return STATUS_SUCCESS; 
	
}