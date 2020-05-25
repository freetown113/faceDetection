#include"TRuntime.h"

using namespace gpuprocess;
using namespace nvinfer1;

TRuntime::TRuntime(NvInferPtr<nvinfer1::ICudaEngine> &&engine, int batch) :
	_engine(std::move(engine)), batch_size(batch)
{
	_context = NvInferPtr<IExecutionContext>(_engine->createExecutionContext());
}

void TRuntime::inference(std::vector<void*>& outputsBuffers)
{
	//print_version();
	cudaStream_t stream;
	CUDA_SAFE_CALL(cudaStreamCreate(&stream));
	//_context->enqueue(batch_size, outputsBuffers.data(), stream, nullptr);
	_context->enqueueV2(outputsBuffers.data(), stream, nullptr);
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
}

bool TRuntime::saveEngine(const char *path)
{
	bool flag = false;
	IHostMemory* serialized_engine = _engine->serialize(); 
	std::ofstream stream(path, std::ios::binary);
	if (stream)
	{
		stream.write(static_cast<char*>(serialized_engine->data()), serialized_engine->size());
		stream.close();
		flag = true;
	}
	return flag;
}