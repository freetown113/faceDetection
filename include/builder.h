#pragma once
#include"netbuilder.h"
#include"cuda_includes.h"

using namespace gpuprocess;

class BuilderFace : public NetBuilder
{
public:
	~BuilderFace();
	NvInferPtr<ICudaEngine> load_onnx_network(const char* model_path, uint32_t max_batch_size) override;
};

class BuilderDistinguish : public NetBuilder
{
public:
	~BuilderDistinguish();
	NvInferPtr<ICudaEngine> load_onnx_network(const char* model_path, uint32_t max_batch_size) override;
};
