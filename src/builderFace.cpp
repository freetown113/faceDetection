#include"builder.h"

using namespace gpuprocess;

BuilderFace::~BuilderFace()
{}

NvInferPtr<ICudaEngine> BuilderFace::load_onnx_network(const char* model_path, uint32_t max_batch_size)
{
	
	IBuilder* builder = createInferBuilder(_logger);

	//NvInferPtr<INetworkDefinition> network(builder->createNetwork());
	//~~~~~~~~~~~~
	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
	//~~~~~~~~~~~~
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, _logger);
	int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;

	if (!parser->parseFromFile(model_path, verbosity))
	{
		for (int i = 0; i < parser->getNbErrors(); ++i)
			std::cout << parser->getError(i)->desc() << std::endl;
	 	std::string msg("failed to parse onnx file");
	 	_logger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
	 	exit(EXIT_FAILURE);
	}

	builder->setMaxBatchSize(max_batch_size);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 9);

	builder->setMaxWorkspaceSize(prop.totalGlobalMem / 4);
	//builder->setMaxWorkspaceSize(16 << 24);

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

	const std::string pluginName = "mGridAnchor";
	const std::string pluginVersion = "2";
	std::vector<nvinfer1::PluginField> f;
	
	float min_size = 32.0;
	float max_size = 64.0;
	int fmShape[2] = { 20, 20 };
	float aspectRatios[1] = { 1 };
	float variance[4] = { 0.1, 0.1, 0.2, 0.2 };
	int numLayer = 1;
	f.emplace_back("minSize", &min_size, PluginFieldType::kFLOAT32, 1);
	f.emplace_back("maxSize", &max_size, PluginFieldType::kFLOAT32, 1);
	f.emplace_back("aspectRatios", aspectRatios, PluginFieldType::kFLOAT32, 1);
	f.emplace_back("featureMapShapes", fmShape, PluginFieldType::kINT32, 1);
	f.emplace_back("variance", variance, PluginFieldType::kFLOAT32, 4);
	f.emplace_back("numLayers", &numLayer, PluginFieldType::kINT32, 1);
	
	nvinfer1::PluginFieldCollection* coll_1 = new nvinfer1::PluginFieldCollection();
	coll_1->nbFields = f.size();
	coll_1->fields = f.data();
	
	mGridAnchorPluginCreator* creator = new mGridAnchorPluginCreator;
	getPluginRegistry()->registerCreator(*creator, "myCreator");
	IPluginV2Ext* plugin = creator->createPlugin("mGridAnchor", coll_1);
	
	std::vector<nvinfer1::ITensor*> input;
	input.resize(2);

	input[0] = network->getOutput(0);
	input[1] = network->getInput(0);

	auto layer1 = network->addPluginV2(input.data(), 2, *plugin);

	std::vector<nvinfer1::PluginField> f2;
	float min_size2 = 64.0;
	float max_size2 = 128.0;
	int fmShape2[2] = { 10, 10 };
	f2.emplace_back("minSize", &min_size2, PluginFieldType::kFLOAT32, 1);
	f2.emplace_back("maxSize", &max_size2, PluginFieldType::kFLOAT32, 1);
	f2.emplace_back("aspectRatios", aspectRatios, PluginFieldType::kFLOAT32, 1);
	f2.emplace_back("featureMapShapes", fmShape2, PluginFieldType::kINT32, 1);
	f2.emplace_back("variance", variance, PluginFieldType::kFLOAT32, 4);
	f2.emplace_back("numLayers", &numLayer, PluginFieldType::kINT32, 1);
	coll_1->nbFields = f2.size();
	coll_1->fields = f2.data();
	IPluginV2Ext* plugin2 = creator->createPlugin("mGridAnchor", coll_1);
	
	input[0] = network->getOutput(1);	
	auto layer2 = network->addPluginV2(input.data(), 2, *plugin2);

	std::vector<nvinfer1::PluginField> f3;
	float min_size3 = 128.0;
	float max_size3 = 256.0;
	int fmShape3[2] = { 5, 5 };
	f3.emplace_back("minSize", &min_size3, PluginFieldType::kFLOAT32, 1);
	f3.emplace_back("maxSize", &max_size3, PluginFieldType::kFLOAT32, 1);
	f3.emplace_back("aspectRatios", aspectRatios, PluginFieldType::kFLOAT32, 1);
	f3.emplace_back("featureMapShapes", fmShape3, PluginFieldType::kINT32, 1);
	f3.emplace_back("variance", variance, PluginFieldType::kFLOAT32, 4);
	f3.emplace_back("numLayers", &numLayer, PluginFieldType::kINT32, 1);
	coll_1->nbFields = f3.size();
	coll_1->fields = f3.data();
	IPluginV2Ext* plugin3 = creator->createPlugin("mGridAnchor", coll_1);
	
	input[0] = network->getOutput(2);
	auto layer3 = network->addPluginV2(input.data(), 2, *plugin3);

	input.resize(3);
	input[0] = layer1->getOutput(0);
	input[1] = layer2->getOutput(0);
	input[2] = layer3->getOutput(0);

	std::cout << "Concat" << std::endl;
	for (int i = 0; i < 3; ++i)
	{
		Dims dim = input[i]->getDimensions();
		std::cout << "Dims: " << dim.nbDims << std::endl;
		for (int j = 0; j < dim.nbDims; j++)
			std::cout << "Dim[" << j << "]: " << dim.d[j] << std::endl;
	}

	auto concat = network->addConcatenation(input.data(), 3);
	concat->setAxis(2);

	nbOutputs = concat->getNbOutputs();
	std::cout << "concat outputs fin: " << nbOutputs << std::endl;
	for (int i = 0; i < nbOutputs; i++)
	{
		Dims dim = concat->getOutput(i)->getDimensions();
		std::cout << "Dims: " << dim.nbDims << std::endl;
		for (int j = 0; j < dim.nbDims; j++)
			std::cout << "Dim[" << j << "]: " << dim.d[j] << std::endl;
	}

	std::vector<nvinfer1::PluginField> p;
	int shareLoc = true;
	int varianceInTarget = false;
	int backgroundLabel = 0;
	int numClasses = 2;
	int topK = 400;
	int keepTopK = 200;
	float confTreshold = 0.1f;
	float nmsTreshold = 0.45f;
	nvinfer1::plugin::CodeTypeSSD code = nvinfer1::plugin::CodeTypeSSD::CENTER_SIZE;
	int inputOrder[4] = { 0,1,2,3 };
	int sigmoid = false;
	int isNormalized = true;
	p.emplace_back(PluginField("shareLocation", &shareLoc, PluginFieldType::kINT32, 1));
	p.emplace_back(PluginField("varianceEncodedInTarget", &varianceInTarget, PluginFieldType::kINT32, 1));
	p.emplace_back(PluginField("backgroundLabelId", &backgroundLabel, PluginFieldType::kINT32, 1));
	p.emplace_back(PluginField("numClasses", &numClasses, PluginFieldType::kINT32, 1));
	p.emplace_back(PluginField("topK", &topK, PluginFieldType::kINT32, 1));
	p.emplace_back(PluginField("keepTopK", &keepTopK, PluginFieldType::kINT32, 1));
	p.emplace_back(PluginField("confidenceThreshold", &confTreshold, PluginFieldType::kFLOAT32, 1));
	p.emplace_back(PluginField("nmsThreshold", &nmsTreshold, PluginFieldType::kFLOAT32, 1));
	p.emplace_back(PluginField("inputOrder", inputOrder, PluginFieldType::kINT32, 4));
	p.emplace_back(PluginField("confSigmoid", &sigmoid, PluginFieldType::kINT32, 1));
	p.emplace_back(PluginField("isNormalized", &isNormalized, PluginFieldType::kINT32, 1));
	p.emplace_back(PluginField("codeType", &code, PluginFieldType::kINT32, 1));

	nvinfer1::PluginFieldCollection* det = new nvinfer1::PluginFieldCollection();
	det->nbFields = p.size();
	det->fields = p.data();

	mNMSPluginCreator* creatorDet = new mNMSPluginCreator;
	getPluginRegistry()->registerCreator(*creatorDet, "myCreator");
	IPluginV2Ext* detector = creatorDet->createPlugin("NMS", det);

	input.resize(4);
	input[0] = network->getOutput(5);
	input[1] = network->getOutput(3);
	input[2] = network->getOutput(4);
	input[3] = concat->getOutput(0);

	auto detectionLayer = network->addPluginV2(input.data(), 4, *detector);
	
	std::cout << "Before transformation " << network->getNbOutputs() << std::endl;
	nbOutputs = network->getNbOutputs();
	std::vector<nvinfer1::ITensor *> tensors = { network->getOutput(0), network->getOutput(1),
		network->getOutput(2), network->getOutput(3), network->getOutput(4), network->getOutput(5) };
	for (int i = 0; i < nbOutputs; i++)
	{
		network->unmarkOutput(*tensors[i]);
	}

	nvinfer1::ITensor* output0 = detectionLayer->getOutput(0); //!!!!!!!!!!!!!!!!!!!!!!!!
	nvinfer1::ITensor* output1 = detectionLayer->getOutput(1); //!!!!!!!!!!!!!!!!!!!!!!!!
	network->markOutput(*output0);
	network->markOutput(*output1);
	std::cout << "After transformation " << network->getNbOutputs() << std::endl;

	nbOutputs = network->getNbOutputs();
	std::cout << "ONNX outputs fin: " << nbOutputs << std::endl;
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
	for (int i = 0; i < _engine->getNbBindings(); i++) 
	{
		if (!_engine->bindingIsInput(i))
			count++;
	}

	plugin->destroy();
	plugin2->destroy();
	plugin3->destroy();
	detector->destroy();
	parser->destroy();
	network->destroy();
	builder->destroy();

	return _engine;
}
