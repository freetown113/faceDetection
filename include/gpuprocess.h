#pragma once
#include <iostream>
#include <fstream>
//#include <iterator>
#include <vector>
//#include "pluginimplement.h"
//#include <cuda_runtime.h>
//#include <npp.h>
//#include <cuda.h>
//#include <nppi.h>
//#include <nppi_geometry_transforms.h>
//#include <nppi_data_exchange_and_initialization.h>
//#include <NvInfer.h>
//#include <NvOnnxParser.h>

namespace gpuprocess
{	
	class Blob
	{
	public:
		Blob(int h, int w, int ch, int el_size);
		Blob(void* buffer, int h, int w, int ch, int el_size);
		Blob() = default;
		~Blob();
		Blob(Blob const& copy);
		Blob(Blob&& copy);
		Blob& operator=(Blob copy);
		friend void swap(Blob& one, Blob& another);
		int getWidth() { return width; };
		int getHeight() { return height; };
		int getElemSize() { return element_size; };
		int getChannels() { return channels; };
		void* getDataPtr() { return data; };		
		int getBufferSize() { return width * height * channels; };
	private:
		void *data;
		int width;
		int height;
		int channels;
		int element_size;
		bool ownership;
	};	

	void resize(Blob& input, Blob& resized);
	void to_planar(Blob& packed, Blob& planar);
	void to_packed(Blob& planar, Blob& packed);
	void channel_swap(Blob& packed);
	void multiply(Blob& input, float values[3]);
	void u8_to_f32(Blob& input, Blob& converted);
	void f32_to_u8(Blob& input, Blob& converted);
	void subtract(Blob& input, float values[3]);
	void cpu_to_gpu(void* host_buffer, Blob& device_buffer);
	void gpu_to_cpu(Blob& device_buffer, void* host_buffer);
		   	 
	class ITRTRuntime
	{
	public:	
		virtual void inference(std::vector<void*>& outputsBuffers) = 0;
		virtual bool saveEngine(const char *path) = 0;
	};
		
	class INetBuilder
	{
	public:
		enum LoadMode
		{
			LOAD_FROM_MODEL,
			LOAD_FROM_ENGINE
		};
	public:		
		virtual ITRTRuntime* create(LoadMode source, const char *path, int batch_size) = 0;
	};

	INetBuilder* getNetBuilder(const char* species);
}


