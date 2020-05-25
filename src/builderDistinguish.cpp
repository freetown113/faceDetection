#include"builder.h"

using namespace gpuprocess;

BuilderDistinguish::~BuilderDistinguish()
{}

NvInferPtr<ICudaEngine> BuilderDistinguish::load_onnx_network(const char* model_path, uint32_t max_batch_size)
{
	NvInferPtr<IBuilder> builder(createInferBuilder(_logger));
	//NvInferPtr<INetworkDefinition> network(builder->createNetwork());
	//~~~~~~~~~~~~
	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	NvInferPtr<INetworkDefinition> network(builder->createNetworkV2(explicitBatch));
	//~~~~~~~~~~~~
	NvInferPtr<nvonnxparser::IParser>parser(nvonnxparser::createParser(*network, _logger));

	int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;

	if (!parser->parseFromFile(model_path, verbosity))
	{
		std::string msg("failed to parse onnx file");
		_logger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
		exit(EXIT_FAILURE);
	}

	builder->setMaxBatchSize(max_batch_size);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	builder->setMaxWorkspaceSize(prop.totalGlobalMem / 4);

	//bool use_fp16 = builder->platformHasFastFp16();
	bool use_fp16 = false;

	DataType model_data_type = use_fp16 ? DataType::kHALF : DataType::kFLOAT;

	if (use_fp16)
	{
		std::cout << "Using FP16 mode" << std::endl;
		builder->setFp16Mode(true);
	}
	int nbLayers = network->getNbLayers();
	
	int nbOutputs = network->getNbOutputs();
	std::cout << "ONNX outputs: " << nbOutputs << std::endl;
	for (int i = 0; i < nbOutputs; i++)
	{
		Dims dim = network->getOutput(i)->getDimensions();
		std::cout << "Dims: " << dim.nbDims << std::endl;
		for (int j = 0; j < dim.nbDims; j++)
			std::cout << "Dim[" << j << "]: " << dim.d[j] << std::endl;
	}
	
	NvInferPtr<ICudaEngine> _engine(builder->buildCudaEngine(*network));
	assert(_engine);

	int count = 0;
	for (int i = 0; i < _engine->getNbBindings(); i++) {
		if (!_engine->bindingIsInput(i))
			count++;
	}

	return _engine;
}