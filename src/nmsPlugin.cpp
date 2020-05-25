#include "nmsPlugin.h"
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::DetectionOutput;
using nvinfer1::plugin::DetectionOutputParameters;
using nvinfer1::plugin::mNMSPluginCreator;

size_t detectionInferenceWorkspaceSize(bool shareLocation, int N, int C1, int C2, int numClasses, int numPredsPerClass,
	int topK, DataType DT_BBOX, DataType DT_SCORE)
{
	size_t wss[7];
	wss[0] = detectionForwardBBoxDataSize(N, C1, DT_BBOX);
	wss[1] = detectionForwardBBoxPermuteSize(shareLocation, N, C1, DT_BBOX);
	wss[2] = detectionForwardPreNMSSize(N, C2);
	wss[3] = detectionForwardPreNMSSize(N, C2);
	wss[4] = detectionForwardPostNMSSize(N, numClasses, topK);
	wss[5] = detectionForwardPostNMSSize(N, numClasses, topK);
	wss[6] = std::max(sortScoresPerClassWorkspaceSize(N, numClasses, numPredsPerClass, DT_SCORE),
		sortScoresPerImageWorkspaceSize(N, numClasses * topK, DT_SCORE));
	return calculateTotalWorkspaceSize(wss, 7);
}

namespace
{
	const char* NMS_PLUGIN_VERSION{ "2" };
	const char* NMS_PLUGIN_NAME{ "mNMS" };
} // namespace

PluginFieldCollection mNMSPluginCreator::mFC{};
std::vector<PluginField> mNMSPluginCreator::mPluginAttributes;

// Constrcutor
DetectionOutput::DetectionOutput(DetectionOutputParametersPP params)
	: param(params)
{
}

DetectionOutput::DetectionOutput(DetectionOutputParametersPP params, int C0, int C1, int C2, int numPriors)
	: param(params)
	, C0(C0)
	, C1(C1)
	, C2(C2)
	, numPriors(numPriors)
{
}

// Parameterized constructor
DetectionOutput::DetectionOutput(const void* data, size_t length)
{
	std::cout << "DetectionOutput" << std::endl;
	const char *d = reinterpret_cast<const char*>(data), *a = d;
	param = read<DetectionOutputParametersPP>(d);
	// Channel size of the locData tensor
	// numPriors * numLocClasses * 10
	C0 = read<int>(d);
	// Channel size of the locData tensor
	// numPriors * numLocClasses * 4
	C1 = read<int>(d);
	// Channel size of the confData tensor
	// numPriors * param.numClasses
	C2 = read<int>(d);
	// Number of bounding boxes per sample
	numPriors = read<int>(d);
	ASSERT(d == a + length);
}

int DetectionOutput::getNbOutputs() const
{
	std::cout << "getNbOutputs" << std::endl;
	// Plugin layer has 3 outputs
	return 3;
}

int DetectionOutput::initialize()
{	
	std::cout << "initialize" << std::endl;
	return STATUS_SUCCESS;
}

void DetectionOutput::terminate() {}

// Returns output dimensions at given index
Dims DetectionOutput::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
	std::cout << "getOutputDimensions" << std::endl;
	ASSERT(nbInputDims == 4);
	ASSERT(index == 0 || index == 1 || index == 2);
	// Output dimensions
	// index 0 : Dimensions 1x param.keepTopK x 13
	// index 1 : Dimensions 1x param.keepTopK x 7
	// index 2: Dimensions 1x1x1
	if (index == 0)
	{
		return DimsCHW(1, param.keepTopK, 13);
	}
	if (index == 1)
	{
		return DimsCHW(1, param.keepTopK, 7);
	}
	return DimsCHW(1, 1, 1);
}

// Returns the workspace size
size_t DetectionOutput::getWorkspaceSize(int maxBatchSize) const
{
	std::cout << "getWorkspaceSize" << std::endl;
	size_t points = detectionInferenceWorkspaceSize(param.shareLocation, maxBatchSize, C0, C2, param.numClasses, numPriors,
		param.topK, DataType::kFLOAT, DataType::kFLOAT);
	size_t rects = detectionInferenceWorkspaceSize(param.shareLocation, maxBatchSize, C1, C2, param.numClasses, numPriors,
		param.topK, DataType::kFLOAT, DataType::kFLOAT);
	return rects + points;
}

// Plugin layer implementation
int DetectionOutput::enqueue(
	int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
	std::cout << "enqueue" << std::endl;
	// Input order {loc, conf, prior}
	const void* const locDataPoints = inputs[param.inputOrder[0]];
	const void* const locDataRects = inputs[param.inputOrder[1]];
	const void* const confData = inputs[param.inputOrder[2]];
	const void* const priorData = inputs[param.inputOrder[3]];

	// Output from plugin index 0: topDetections index 1: keepCount
	void* topDetectionsPoints = outputs[0];
	void* topDetectionsRects = outputs[1];
	void* keepCount = outputs[2];

	const int offset = detectionInferenceWorkspaceSize(param.shareLocation, 1, C1, C2, param.numClasses, numPriors,
		param.topK, DataType::kFLOAT, DataType::kFLOAT);

	pluginStatus_t status = detectionInference(stream, batchSize, C0, C1, C2, param.shareLocation,
		param.varianceEncodedInTarget, param.backgroundLabelId, numPriors, param.numClasses, param.topK, param.keepTopK,
		param.confidenceThreshold, param.nmsThreshold, param.codeType, DataType::kFLOAT, locDataPoints, offset, locDataRects, priorData,
		DataType::kFLOAT, confData, keepCount, topDetectionsPoints, topDetectionsRects, workspace, param.isNormalized, param.confSigmoid);
	ASSERT(status == STATUS_SUCCESS);
	return 0;
}

// Returns the size of serialized parameters
size_t DetectionOutput::getSerializationSize() const
{
	std::cout << "getSerializationSize" << std::endl;
	// DetectionOutputParameters, C0,C1,C2,numPriors
	return sizeof(DetectionOutputParametersPP) + sizeof(int) * 4;
}

// Serialization of plugin parameters
void DetectionOutput::serialize(void* buffer) const
{
	std::cout << "serialize" << std::endl;
	char *d = reinterpret_cast<char*>(buffer), *a = d;
	write(d, param);
	write(d, C0);
	write(d, C1);
	write(d, C2);
	write(d, numPriors);
	ASSERT(d == a + getSerializationSize());
}

// Check if the DataType and Plugin format is supported
bool DetectionOutput::supportsFormat(DataType type, PluginFormat format) const
{
	std::cout << "supportsFormat" << std::endl;
	return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

// Get the plugin type
const char* DetectionOutput::getPluginType() const
{
	std::cout << "getPluginType" << std::endl;
	return NMS_PLUGIN_NAME;
}

// Get the plugin version
const char* DetectionOutput::getPluginVersion() const
{
	std::cout << "getPluginVersion" << std::endl;
	return NMS_PLUGIN_VERSION;
}

// Clean up
void DetectionOutput::destroy()
{
	std::cout << "destroy" << std::endl;
	delete this;
}

// Cloning the plugin
IPluginV2Ext* DetectionOutput::clone() const
{
	std::cout << "clone" << std::endl;
	// Create a new instance
	IPluginV2Ext* plugin = new DetectionOutput(param, C0, C1, C2, numPriors);

	// Set the namespace
	plugin->setPluginNamespace(mPluginNamespace);
	return plugin;
}

// Set plugin namespace
void DetectionOutput::setPluginNamespace(const char* pluginNamespace)
{
	std::cout << "setPluginNamespace" << std::endl;
	mPluginNamespace = pluginNamespace;
}

const char* DetectionOutput::getPluginNamespace() const
{
	std::cout << "getPluginNamespace" << std::endl;
	return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index.
DataType DetectionOutput::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
	std::cout << "getOutputDataTypeNMS " << index << std::endl;
	// Two outputs
	ASSERT(index == 0 || index == 1 || index == 2);
	return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool DetectionOutput::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
	std::cout << "isOutputBroadcastAcrossBatch" << std::endl;
	return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool DetectionOutput::canBroadcastInputAcrossBatch(int inputIndex) const
{
	std::cout << "canBroadcastInputAcrossBatch" << std::endl;
	return false;
}

// Configure the layer with input and output data types.
// inutDims: input Dimensions for the plugin layer
// nInputs : Number of inputs to the plugin layer
// outputDims: output Dimensions from the plugin layer
// nOutputs: number of outputs from the plugin layer
// type: DataType configuration for the plugin layer
// format: format NCHW, NHWC etc
// maxbatchSize: maximum batch size for the plugin layer
void DetectionOutput::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
	const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
	const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
	std::cout << "configurePlugin" << std::endl;
	// Number of input dimension should be 3
	ASSERT(nbInputs == 4);

	// Number of output dimension wil be 2
	ASSERT(nbOutputs == 3);

	// Verify all the input dimensions
	for (int i = 0; i < nbInputs; i++)
	{
		ASSERT(inputDims[i].nbDims == 3);
	}

	// Verify all the output dimensions
	for (int i = 0; i < nbOutputs; i++)
	{
		ASSERT(outputDims[i].nbDims == 3);
	}

	// Configure C1, C2 and numPriors
	// Input ordering  C1, C2, numPriors
	C0 = inputDims[param.inputOrder[0]].d[0];
	C1 = inputDims[param.inputOrder[1]].d[0];
	C2 = inputDims[param.inputOrder[2]].d[0];

	const int nbBoxCoordinates = 4;
	std::cout << "nbBoxCoordinates_test " << nbBoxCoordinates << std::endl;
	numPriors = inputDims[param.inputOrder[3]].d[1] / nbBoxCoordinates;
	const int numLocClasses = param.shareLocation ? 1 : param.numClasses;

	// Verify C0
	ASSERT(numPriors * numLocClasses * 10 == inputDims[param.inputOrder[0]].d[0]);

	// Verify C1
	ASSERT(numPriors * numLocClasses * nbBoxCoordinates == inputDims[param.inputOrder[1]].d[0]);
	
	// Verify C2
	ASSERT(numPriors * param.numClasses == inputDims[param.inputOrder[2]].d[0]);
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void DetectionOutput::attachToContext(
	cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void DetectionOutput::detachFromContext() {}

// Plugin creator constructor
mNMSPluginCreator::mNMSPluginCreator()
{
	std::cout << "mNMSPluginCreator" << std::endl;
	// NMS Plugin field meta data {name,  data, type, length}
	mPluginAttributes.emplace_back(PluginField("shareLocation", nullptr, PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("varianceEncodedInTarget", nullptr, PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("backgroundLabelId", nullptr, PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("numClasses", nullptr, PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("topK", nullptr, PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("keepTopK", nullptr, PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("confidenceThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
	mPluginAttributes.emplace_back(PluginField("nmsThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
	mPluginAttributes.emplace_back(PluginField("inputOrder", nullptr, PluginFieldType::kINT32, 4));
	mPluginAttributes.emplace_back(PluginField("confSigmoid", nullptr, PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("isNormalized", nullptr, PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("codeType", nullptr, PluginFieldType::kINT32, 1));

	mFC.nbFields = mPluginAttributes.size();
	mFC.fields = mPluginAttributes.data();
}

// Returns the plugin name
const char* mNMSPluginCreator::getPluginName() const
{
	std::cout << "getPluginName" << std::endl;
	return NMS_PLUGIN_NAME;
}

// Returns the plugin version
const char* mNMSPluginCreator::getPluginVersion() const
{
	std::cout << "getPluginVersion" << std::endl;
	return NMS_PLUGIN_VERSION;
}

// Returns the plugin field names
const PluginFieldCollection* mNMSPluginCreator::getFieldNames()
{
	std::cout << "mNMSPluginCreator" << std::endl;
	return &mFC;
}

// Creates the NMS plugin
IPluginV2Ext* mNMSPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
	std::cout << "createPlugin" << std::endl;
	const PluginField* fields = fc->fields;
	// Default init values for TF SSD network
	params.codeType = CodeTypeSSD::TF_CENTER; // There is no implementation for TF_CENTER
	params.inputOrder[0] = 0;
	params.inputOrder[1] = 1;
	params.inputOrder[2] = 2;
	params.inputOrder[3] = 3;

	// Read configurations from  each fields
	for (int i = 0; i < fc->nbFields; ++i)
	{
		const char* attrName = fields[i].name;
		if (!strcmp(attrName, "shareLocation"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			params.shareLocation = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "varianceEncodedInTarget"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			params.varianceEncodedInTarget = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "backgroundLabelId"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			params.backgroundLabelId = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "numClasses"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			params.numClasses = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "topK"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			params.topK = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "keepTopK"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			params.keepTopK = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "confidenceThreshold"))
		{
			ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
			params.confidenceThreshold = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "nmsThreshold"))
		{
			ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
			params.nmsThreshold = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "confSigmoid"))
		{
			params.confSigmoid = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "isNormalized"))
		{
			params.isNormalized = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "inputOrder"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			const int size = fields[i].length;
			const int* o = static_cast<const int*>(fields[i].data);
			for (int j = 0; j < size; j++)
			{
				params.inputOrder[j] = *o;
				o++;
			}
		}
		else if (!strcmp(attrName, "codeType"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			params.codeType = static_cast<CodeTypeSSD>(*(static_cast<const int*>(fields[i].data)));
		}
	}

	DetectionOutput* obj = new DetectionOutput(params);
	obj->setPluginNamespace(mNamespace.c_str());
	return obj;
}

IPluginV2Ext* mNMSPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
	std::cout << "deserializePlugin" << std::endl;
	// This object will be deleted when the network is destroyed, which will
	// call NMS::destroy()
	DetectionOutput* obj = new DetectionOutput(serialData, serialLength);
	obj->setPluginNamespace(mNamespace.c_str());
	return obj;
}
