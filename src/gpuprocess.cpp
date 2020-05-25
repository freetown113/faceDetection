#include"gpuprocess.h"
#include"cuda_includes.h"


using namespace gpuprocess;

void gpuprocess::resize(Blob& input, Blob& resized)
{
	if (input.getElemSize() == resized.getElemSize() && input.getElemSize() == sizeof(uint8_t))
	{
		auto stat = nppiResize_8u_C3R(static_cast<Npp8u*>(input.getDataPtr()), input.getWidth() * 3, { input.getWidth(), input.getHeight() },
			{ 0, 0, input.getWidth(), input.getHeight() }, static_cast<Npp8u*>(resized.getDataPtr()), resized.getWidth() * 3,
			{ resized.getWidth(), resized.getHeight() }, { 0, 0, resized.getWidth(), resized.getHeight() }, NPPI_INTER_LANCZOS);
	}
	else if (input.getElemSize() == resized.getElemSize() && input.getElemSize() == sizeof(float))
	{
		auto stat = nppiResize_32f_C3R(static_cast<Npp32f*>(input.getDataPtr()), input.getWidth() * 3 * sizeof(float),
			{ input.getWidth(), input.getHeight() }, { 0, 0, input.getWidth(), input.getHeight() },
			static_cast<Npp32f*>(resized.getDataPtr()), resized.getWidth() * 3 * sizeof(float), { resized.getWidth(), resized.getHeight() },
			{ 0, 0, resized.getWidth(), resized.getHeight() }, NPPI_INTER_LANCZOS);
	}
	else
		std::cerr << "The library is anticipated to be used with only two types: 8bits and 32bits!" << std::endl;
}

void gpuprocess::to_planar(Blob& packed, Blob& planar)
{	
	if (packed.getElemSize() == planar.getElemSize() && packed.getElemSize() == sizeof(uint8_t))
	{
		Npp8u* blue = static_cast<Npp8u*>(planar.getDataPtr());
		Npp8u* green = blue + planar.getHeight() * planar.getWidth();
		Npp8u* red = green + planar.getHeight() * planar.getWidth();
		Npp8u* channels[3] = { blue, green, red };

		auto transpose = nppiCopy_8u_C3P3R(static_cast<Npp8u*>(packed.getDataPtr()), packed.getWidth() * 3, channels,
			planar.getWidth(), { planar.getWidth(), planar.getHeight() });
	}
	else if (packed.getElemSize() == planar.getElemSize() && packed.getElemSize() == sizeof(float))
	{
		Npp32f* blue = static_cast<Npp32f*>(planar.getDataPtr());
		Npp32f* green = blue + planar.getHeight() * planar.getWidth();
		Npp32f* red = green + planar.getHeight() * planar.getWidth();
		Npp32f* channels[3] = { blue, green, red };

		auto transpose = nppiCopy_32f_C3P3R(static_cast<Npp32f*>(packed.getDataPtr()), packed.getWidth() * 3
			* sizeof(float), channels, planar.getWidth() * sizeof(float), { planar.getWidth(), planar.getHeight() });
	}
	else
		std::cerr << "The library is anticipated to be used with only two types: 8bits and 32bits!" << std::endl;
}

void gpuprocess::to_packed(Blob& planar, Blob& packed)
{
	if (packed.getElemSize() == planar.getElemSize() && packed.getElemSize() == sizeof(uint8_t))
	{
		Npp8u* blue = static_cast<Npp8u*>(planar.getDataPtr());
		Npp8u* green = blue + planar.getHeight() * planar.getWidth();
		Npp8u* red = green + planar.getHeight() * planar.getWidth();
		Npp8u* channels[3] = { blue, green, red };

		auto transpose = nppiCopy_8u_P3C3R(channels, planar.getWidth(), static_cast<Npp8u*>(packed.getDataPtr()),
			packed.getWidth() * 3, { packed.getWidth(), packed.getHeight() });
	}
	else if (packed.getElemSize() == planar.getElemSize() && packed.getElemSize() == sizeof(float))
	{
		Npp32f* blue = static_cast<Npp32f*>(planar.getDataPtr());
		Npp32f* green = blue + planar.getHeight() * planar.getWidth();
		Npp32f* red = green + planar.getHeight() * planar.getWidth();
		Npp32f* channels[3] = { blue, green, red };

		auto transpose = nppiCopy_32f_P3C3R(channels, planar.getWidth() * sizeof(float), static_cast<Npp32f*>(packed.getDataPtr()),
			packed.getWidth() * 3 * sizeof(float), { packed.getWidth(), packed.getHeight() });
	}
	else
		std::cerr << "The library is anticipated to be used with only two types: 8bits and 32bits!" << std::endl;
}

void gpuprocess::channel_swap(Blob& input)
{
	const int order[] = { 2, 1, 0 };
	if (input.getElemSize() == sizeof(uint8_t))
	{	
		auto channelSwap = nppiSwapChannels_8u_C3IR(static_cast<Npp8u*>(input.getDataPtr()), input.getWidth() * 3,
			{ input.getWidth(), input.getHeight() }, order);
	}
	else if(input.getElemSize() == sizeof(float))
	{
		auto channelSwap = nppiSwapChannels_32f_C3IR(static_cast<Npp32f*>(input.getDataPtr()), input.getWidth() * 3 
			* sizeof(float), { input.getWidth(), input.getHeight() }, order);
	}
}
//32fc3 buffer's size is w * h * c * sizeof(float)
void gpuprocess::u8_to_f32(Blob& input, Blob& converted)
{
	auto convert = nppiConvert_8u32f_C3R(static_cast<Npp8u*>(input.getDataPtr()), input.getWidth() * 3,
		static_cast<Npp32f*>(converted.getDataPtr()), converted.getWidth() * 3 * sizeof(float),
		{ input.getWidth(), input.getHeight() });
}
//32fc3 buffer's size is w * h * c * sizeof(float)
void gpuprocess::f32_to_u8(Blob& input, Blob& converted)
{
	auto convert = nppiConvert_32f8u_C3R(static_cast<Npp32f*>(input.getDataPtr()), input.getWidth() * 3 *
		sizeof(float), static_cast<Npp8u*>(converted.getDataPtr()), converted.getWidth() * 3,
		{ input.getWidth(), input.getHeight() }, NPP_RND_NEAR);
}

void gpuprocess::subtract(Blob& input, float values[3])
{
	Npp32f channels[3] = { values[0], values[1], values[2] };
	auto substract = nppiSubC_32f_C3IR(channels, static_cast<Npp32f*>(input.getDataPtr()), input.getWidth() * 3 * sizeof(float),
		{ input.getWidth(),input.getHeight() });	
}

void gpuprocess::multiply(Blob& input, float values[3])
{
	Npp32f channels[3] = { values[0], values[1], values[2] };
	auto div = nppiMulC_32f_C3IR(channels, static_cast<Npp32f*>(input.getDataPtr()), input.getWidth() * 3 * sizeof(float),
		{ input.getWidth(), input.getHeight() });
}

void gpuprocess::cpu_to_gpu(void* host_buffer, Blob& device_buffer)
{
	CUDA_SAFE_CALL(cudaMemcpy(device_buffer.getDataPtr(), host_buffer, device_buffer.getBufferSize()
		* device_buffer.getElemSize(), cudaMemcpyHostToDevice));
}
void gpuprocess::gpu_to_cpu(Blob& device_buffer, void* host_buffer)
{
	CUDA_SAFE_CALL(cudaMemcpy(host_buffer, device_buffer.getDataPtr(), device_buffer.getBufferSize()
		* device_buffer.getElemSize(), cudaMemcpyDeviceToHost));
}

