#include"netbuilder.h"


using namespace gpuprocess;

NetBuilder::NetBuilder()
	: _infer(createInferRuntime(_logger), NvInferDestructor<nvinfer1::IRuntime>())
{
	cudaSetDevice(9);
	//_infer = createInferRuntime(_logger); //_plugin_factory
	//initLibNvInferPlugins(&_logger, "");
}

NvInferPtr<nvinfer1::ICudaEngine> NetBuilder::load_from_engine(const char *path)
{
	std::ifstream in(path, std::ios::binary | std::ios::ate);
	if (!in)
		return nullptr;
	auto buffer_size = in.tellg();
	std::vector<char> buffer(buffer_size);
	in.seekg(std::ios::beg);
	in.read(buffer.data(), buffer.size());
	REGISTER_TENSORRT_PLUGIN(mGridAnchorPluginCreator);
	REGISTER_TENSORRT_PLUGIN(mNMSPluginCreator);
	/*auto ret = _infer->deserializeCudaEngine(
		static_cast<const void *>(buffer.data()), buffer_size, &_plugin_factory);*/
	//_infer->destroy();
	return NvInferPtr<ICudaEngine>(_infer->deserializeCudaEngine(
		static_cast<const void *>(buffer.data()), buffer_size, nullptr));
}

ITRTRuntime* NetBuilder::create(LoadMode source, const char *path, int batch_size = 1)
{		
	NvInferPtr<ICudaEngine> engine = nullptr;
	if (source == LOAD_FROM_MODEL)
	{
		engine = load_onnx_network(path, batch_size);
	}
	else if (source == LOAD_FROM_ENGINE)
	{
		engine = load_from_engine(path);
	}
	TRuntime* instance = new TRuntime(std::move(engine), batch_size);
	return instance;
}
