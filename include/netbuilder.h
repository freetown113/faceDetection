#pragma once

#include <memory>

#include"TRuntime.h"
//#include"pluginimplement.h"

using namespace gpuprocess;


class NetBuilder : public gpuprocess::INetBuilder
{
public:
	NetBuilder();
	virtual ~NetBuilder() = default;	
	NvInferPtr<ICudaEngine> load_from_engine(const char *path);
	ITRTRuntime* create(LoadMode source, const char *path, int batch_size) override;
	virtual NvInferPtr<ICudaEngine> load_onnx_network(const char* model_path, uint32_t max_batch_size) = 0;
	bool load_caffe_network(const char* prototxt_path, const char* model_path,
		const std::vector<std::string>& output_blobs, uint32_t max_batch_size);	
	std::string getEngineBuffer() { return engine_buffer; };
protected:
	Logger _logger;
	NvInferPtr<IRuntime> _infer;
	//PluginFactory _plugin_factory;
	std::string engine_buffer;
};
