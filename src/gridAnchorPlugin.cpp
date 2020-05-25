#include "gridAnchorPlugin.h"
#include <cstring>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::GridAnchorGenerator;
using nvinfer1::plugin::mGridAnchorPluginCreator;

namespace
{
	const char* GRID_ANCHOR_PLUGIN_VERSION{ "2" };
	const char* GRID_ANCHOR_PLUGIN_NAME{ "mGridAnchor" };
} // namespace
PluginFieldCollection mGridAnchorPluginCreator::mFC{};
std::vector<PluginField> mGridAnchorPluginCreator::mPluginAttributes;

nvinfer1::plugin::GridAnchorGenerator::GridAnchorGenerator(const nvinfer1::plugin::GridAnchorParameters* paramIn, int mNumLayers)
	: mNumLayers(mNumLayers)
{
	std::cout << "GridAnchorGenerator_start" << std::endl;
	CUASSERT(cudaMallocHost((void**)&mNumPriors, mNumLayers * sizeof(int)));
	CUASSERT(cudaMallocHost((void**)&mDeviceWidths, mNumLayers * sizeof(Weights)));
	CUASSERT(cudaMallocHost((void**)&mDeviceHeights, mNumLayers * sizeof(Weights)));

	mParam.resize(mNumLayers);
	for (int id = 0; id < mNumLayers; id++)
	{
		mParam[id] = paramIn[id];
		ASSERT(mParam[id].numAspectRatios >= 0 && mParam[id].aspectRatios != nullptr);

		mParam[id].aspectRatios = (float*)malloc(sizeof(float) * mParam[id].numAspectRatios);

		for (int i = 0; i < paramIn[id].numAspectRatios; ++i)
		{
			mParam[id].aspectRatios[i] = paramIn[id].aspectRatios[i];
		}

		for (int i = 0; i < 4; ++i)
		{
			mParam[id].variance[i] = paramIn[id].variance[i];
		}

		std::vector<float> tmpScales(mNumLayers + 1);

		// Calculate the scales of SSD model for each layer
		for (int i = 0; i < mNumLayers; i++)
		{
			std::cout << "GridAnchorGenerator_1 " << mNumLayers - 1 << std::endl;
			tmpScales[i] = (mParam[id].minSize + (mParam[id].maxSize - mParam[id].minSize) * id / (mNumLayers - 1));
			
		}
		// Add another 1.0f to tmpScales to prevent going out side of the vector in calculating the scale_next.
		tmpScales.push_back(1.0f); // has 7 entries
		// scale0 are for the first layer specifically
		std::vector<float> scale0 = { 0.1f, tmpScales[0], tmpScales[0] };

		std::vector<float> aspect_ratios;
		std::vector<float> scales;

		// The first layer is different
		if (id == 0)
		{
			for (int i = 0; i < mParam[id].numAspectRatios; i++)
			{
				aspect_ratios.push_back(mParam[id].aspectRatios[i]);
				scales.push_back(scale0[i]);
			}
			mNumPriors[id] = mParam[id].numAspectRatios;
		}

		else
		{
			for (int i = 0; i < mParam[id].numAspectRatios; i++)
			{
				aspect_ratios.push_back(mParam[id].aspectRatios[i]);
			}
			// Additional aspect ratio of 1.0 as described in the paper
			aspect_ratios.push_back(1.0);

			// scales
			for (int i = 0; i < mParam[id].numAspectRatios; i++)
			{
				scales.push_back(tmpScales[id]);
			}
			auto scale_next = (id == mNumLayers - 1)
				? 1.0
				: (mParam[id].minSize + (mParam[id].maxSize - mParam[id].minSize) * (id + 1) / (mNumLayers - 1));
			scales.push_back(sqrt(tmpScales[id] * scale_next));

			mNumPriors[id] = mParam[id].numAspectRatios + 1;
		}

		std::vector<float> tmpWidths;
		std::vector<float> tmpHeights;
		// Calculate the width and height of the prior boxes
		for (int i = 0; i < mNumPriors[id]; i++)
		{
			float sqrt_AR = sqrt(aspect_ratios[i]);
			std::cout << "GridAnchorGenerator_3: " << sqrt_AR << std::endl;
			tmpWidths.push_back(scales[i] * sqrt_AR);
			tmpHeights.push_back(scales[i] / sqrt_AR);
		}

		mDeviceWidths[id] = copyToDevice(&tmpWidths[0], tmpWidths.size());
		mDeviceHeights[id] = copyToDevice(&tmpHeights[0], tmpHeights.size());
	}
}

nvinfer1::plugin::GridAnchorGenerator::GridAnchorGenerator(const void* data, size_t length)
{
	std::cout << "GridAnchorGenerator2_start" << std::endl;
	const char *d = reinterpret_cast<const char*>(data), *a = d;
	mNumLayers = read<int>(d);
	CUASSERT(cudaMallocHost((void**)&mNumPriors, mNumLayers * sizeof(int)));
	CUASSERT(cudaMallocHost((void**)&mDeviceWidths, mNumLayers * sizeof(Weights)));
	CUASSERT(cudaMallocHost((void**)&mDeviceHeights, mNumLayers * sizeof(Weights)));
	mParam.resize(mNumLayers);
	for (int id = 0; id < mNumLayers; id++)
	{
		// we have to deserialize GridAnchorParameters by hand
		mParam[id].minSize = read<float>(d);
		mParam[id].maxSize = read<float>(d);
		mParam[id].numAspectRatios = read<int>(d);
		mParam[id].aspectRatios = (float*)malloc(sizeof(float) * mParam[id].numAspectRatios);
		for (int i = 0; i < mParam[id].numAspectRatios; ++i)
		{
			mParam[id].aspectRatios[i] = read<float>(d);
		}
		mParam[id].H = read<int>(d);
		mParam[id].W = read<int>(d);
		for (int i = 0; i < 4; ++i)
		{
			mParam[id].variance[i] = read<float>(d);
		}

		mNumPriors[id] = read<int>(d);
		mDeviceWidths[id] = deserializeToDevice(d, mNumPriors[id]);
		mDeviceHeights[id] = deserializeToDevice(d, mNumPriors[id]);
	}

	ASSERT(d == a + length);
}

GridAnchorGenerator::~GridAnchorGenerator()
{
	std::cout << "GridAnchorGenerator_dest_start" << std::endl;
	for (int id = 0; id < mNumLayers; id++)
	{
		CUERRORMSG(cudaFree(const_cast<void*>(mDeviceWidths[id].values)));
		CUERRORMSG(cudaFree(const_cast<void*>(mDeviceHeights[id].values)));
		free(mParam[id].aspectRatios);
	}
	CUERRORMSG(cudaFreeHost(mNumPriors));
	CUERRORMSG(cudaFreeHost(mDeviceWidths));
	CUERRORMSG(cudaFreeHost(mDeviceHeights));
}

int GridAnchorGenerator::getNbOutputs() const
{
	std::cout << "getNbOutputs" << std::endl;
	return mNumLayers;
}

Dims GridAnchorGenerator::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
	std::cout << "getOutputDimensions" << std::endl;
	// Particularity of the PriorBox layer: no batchSize dimension needed
	// 2 channels. First channel stores the mean of each prior coordinate.
	// Second channel stores the variance of each prior coordinate.
	return DimsCHW(2, mParam[index].H * mParam[index].W * mNumPriors[index] * 4, 1);
}

int GridAnchorGenerator::initialize()
{
	std::cout << "initialize" << std::endl;
	return STATUS_SUCCESS;
}

void GridAnchorGenerator::terminate() {}

size_t GridAnchorGenerator::getWorkspaceSize(int maxBatchSize) const
{
	std::cout << "getWorkspaceSize" << std::endl;
	return 0;
}

int GridAnchorGenerator::enqueue(
	int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
	std::cout << "enqueue" << std::endl;
	// Generate prior boxes for each layer
	for (int id = 0; id < mNumLayers; id++)
	{
		void* outputData = outputs[id];
		pluginStatus_t status = anchorGridInference(
			stream, mParam[id], mNumPriors[id], mDeviceWidths[id].values, mDeviceHeights[id].values, outputData);
		ASSERT(status == STATUS_SUCCESS);
	}
	return STATUS_SUCCESS;
}

size_t GridAnchorGenerator::getSerializationSize() const
{
	std::cout << "getSerializationSize" << std::endl;
	size_t sum = sizeof(int); // mNumLayers
	for (int i = 0; i < mNumLayers; i++)
	{
		sum += 4 * sizeof(int); // mNumPriors, mParam[i].{numAspectRatios, H, W}
		sum += (6 + mParam[i].numAspectRatios)
			* sizeof(float); // mParam[i].{minSize, maxSize, aspectRatios, variance[4]}
		sum += mDeviceWidths[i].count * sizeof(float);
		sum += mDeviceHeights[i].count * sizeof(float);
	}
	return sum;
}

void GridAnchorGenerator::serialize(void* buffer) const
{
	std::cout << "serialize" << std::endl;
	char *d = reinterpret_cast<char*>(buffer), *a = d;
	write(d, mNumLayers);
	for (int id = 0; id < mNumLayers; id++)
	{
		// we have to serialize GridAnchorParameters by hand
		write(d, mParam[id].minSize);
		write(d, mParam[id].maxSize);
		write(d, mParam[id].numAspectRatios);
		for (int i = 0; i < mParam[id].numAspectRatios; ++i)
		{
			write(d, mParam[id].aspectRatios[i]);
		}
		write(d, mParam[id].H);
		write(d, mParam[id].W);
		for (int i = 0; i < 4; ++i)
		{
			write(d, mParam[id].variance[i]);
		}

		write(d, mNumPriors[id]);
		serializeFromDevice(d, mDeviceWidths[id]);
		serializeFromDevice(d, mDeviceHeights[id]);
	}
	ASSERT(d == a + getSerializationSize());
}

Weights GridAnchorGenerator::copyToDevice(const void* hostData, size_t count)
{
	std::cout << "copyToDevice" << std::endl;
	void* deviceData;
	CUASSERT(cudaMalloc(&deviceData, count * sizeof(float)));
	CUASSERT(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
	return Weights{ DataType::kFLOAT, deviceData, int64_t(count) };
}

void GridAnchorGenerator::serializeFromDevice(char*& hostBuffer, Weights deviceWeights) const
{
	std::cout << "serializeFromDevice" << std::endl;
	cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost);
	hostBuffer += deviceWeights.count * sizeof(float);
}

Weights GridAnchorGenerator::deserializeToDevice(const char*& hostBuffer, size_t count)
{
	std::cout << "deserializeToDevice" << std::endl;
	Weights w = copyToDevice(hostBuffer, count);
	hostBuffer += count * sizeof(float);
	return w;
}
bool GridAnchorGenerator::supportsFormat(DataType type, PluginFormat format) const
{
	std::cout << "supportsFormat" << std::endl;
	return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

const char* GridAnchorGenerator::getPluginType() const
{
	std::cout << "getPluginType" << std::endl;
	return GRID_ANCHOR_PLUGIN_NAME;
}

const char* GridAnchorGenerator::getPluginVersion() const
{
	std::cout << "getPluginVersion" << std::endl;
	return GRID_ANCHOR_PLUGIN_VERSION;
}

// Set plugin namespace
void GridAnchorGenerator::setPluginNamespace(const char* pluginNamespace)
{
	std::cout << "setPluginNamespace" << std::endl;
	mPluginNamespace = pluginNamespace;
}

const char* GridAnchorGenerator::getPluginNamespace() const
{
	std::cout << "getPluginNamespace" << std::endl;
	return mPluginNamespace;
}

#include <iostream>
// Return the DataType of the plugin output at the requested index
DataType GridAnchorGenerator::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
	std::cout << "getOutputDataTypeGA " << index << " " << mNumLayers << std::endl;
	ASSERT(index < mNumLayers);
	return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool GridAnchorGenerator::isOutputBroadcastAcrossBatch(
	int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
	std::cout << "isOutputBroadcastAcrossBatch" << std::endl;
	return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool GridAnchorGenerator::canBroadcastInputAcrossBatch(int inputIndex) const
{
	std::cout << "canBroadcastInputAcrossBatch" << std::endl;
	return false;
}

// Configure the layer with input and output data types.
void GridAnchorGenerator::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
	const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
	const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
	std::cout << "configurePlugin" << std::endl;
	ASSERT(nbOutputs == mNumLayers);
	ASSERT(outputDims[0].nbDims == 3);
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void GridAnchorGenerator::attachToContext(
	cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator){}

// Detach the plugin object from its execution context.
void GridAnchorGenerator::detachFromContext() {}

void GridAnchorGenerator::destroy()
{
	std::cout << "destroy" << std::endl;
	delete this;
}

IPluginV2Ext* GridAnchorGenerator::clone() const
{
	std::cout << "clone" << std::endl;
	IPluginV2Ext* plugin = new GridAnchorGenerator(mParam.data(), mNumLayers);
	plugin->setPluginNamespace(mPluginNamespace);
	return plugin;
}

mGridAnchorPluginCreator::mGridAnchorPluginCreator()
{
	std::cout << "mGridAnchorPluginCreator" << std::endl;
	mPluginAttributes.emplace_back(PluginField("minSize", nullptr, PluginFieldType::kFLOAT32, 1));
	mPluginAttributes.emplace_back(PluginField("maxSize", nullptr, PluginFieldType::kFLOAT32, 1));
	mPluginAttributes.emplace_back(PluginField("aspectRatios", nullptr, PluginFieldType::kFLOAT32, 1));
	mPluginAttributes.emplace_back(PluginField("featureMapShapes", nullptr, PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("variance", nullptr, PluginFieldType::kFLOAT32, 4));
	mPluginAttributes.emplace_back(PluginField("numLayers", nullptr, PluginFieldType::kINT32, 1));

	mFC.nbFields = mPluginAttributes.size();
	mFC.fields = mPluginAttributes.data();
}

const char* mGridAnchorPluginCreator::getPluginName() const
{
	std::cout << "getPluginName" << std::endl;
	return GRID_ANCHOR_PLUGIN_NAME;
}

const char* mGridAnchorPluginCreator::getPluginVersion() const
{
	std::cout << "getPluginVersion" << std::endl;
	return GRID_ANCHOR_PLUGIN_VERSION;
}

const PluginFieldCollection* mGridAnchorPluginCreator::getFieldNames()
{
	std::cout << "mGridAnchorPluginCreator" << std::endl;
	return &mFC;
}

IPluginV2Ext* mGridAnchorPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{	
	std::cout << "createPlugin" << std::endl;
	float minScale = 0.2F, maxScale = 0.95F;
	int numLayers = 6;
	std::vector<float> aspectRatios;
	std::vector<int> fMapShapes;
	std::vector<float> layerVariances;
	const PluginField* fields = fc->fields;
	for (int i = 0; i < fc->nbFields; ++i)
	{
		const char* attrName = fields[i].name;
		if (!strcmp(attrName, "numLayers"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			numLayers = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "minSize"))
		{
			ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
			minScale = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "maxSize"))
		{
			ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
			maxScale = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "variance"))
		{
			ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
			int size = fields[i].length;
			layerVariances.reserve(size);
			const auto* lVar = static_cast<const float*>(fields[i].data);
			for (int j = 0; j < size; j++)
			{
				layerVariances.push_back(*lVar);
				lVar++;
			}
		}
		else if (!strcmp(attrName, "aspectRatios"))
		{
			ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
			int size = fields[i].length;
			aspectRatios.reserve(size);
			const auto* aR = static_cast<const float*>(fields[i].data);
			for (int j = 0; j < size; j++)
			{
				aspectRatios.push_back(*aR);
				aR++;
			}
		}
		else if (!strcmp(attrName, "featureMapShapes"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			int size = fields[i].length;
			fMapShapes.reserve(size);
			const int* fMap = static_cast<const int*>(fields[i].data);
			for (int j = 0; j < size; j++)
			{
				fMapShapes.push_back(*fMap);
				fMap++;
			}
		}
	}
	// Reducing the number of boxes predicted by the first layer.
	// This is in accordance with the standard implementation.
	std::vector<float> firstLayerAspectRatios;
	std::cout << "Feature map " << (int)fMapShapes.size() << std::endl;
	ASSERT(numLayers > 0);
	ASSERT((int)fMapShapes.size() == numLayers);

	int numFirstLayerARs = 1;
	// First layer only has the first 3 aspect ratios from aspectRatios
	firstLayerAspectRatios.reserve(numFirstLayerARs);
	for (int i = 0; i < numFirstLayerARs; ++i)
	{
		firstLayerAspectRatios.push_back(aspectRatios[i]);
	}
	// A comprehensive list of box parameters that are required by anchor generator
	std::vector<GridAnchorParameters> boxParams(numLayers);

	// One set of box parameters for one layer
	for (int i = 0; i < numLayers; i++)
	{
		// Only the first layer is different
		if (i == 0)
		{
			boxParams[i] = { minScale, maxScale, firstLayerAspectRatios.data(), (int)firstLayerAspectRatios.size(),
				fMapShapes[i], fMapShapes[i],
				{layerVariances[0], layerVariances[1], layerVariances[2], layerVariances[3]} };
		}
		else
		{
			boxParams[i] = { minScale, maxScale, aspectRatios.data(), (int)aspectRatios.size(), fMapShapes[i],
				fMapShapes[i], {layerVariances[0], layerVariances[1], layerVariances[2], layerVariances[3]} };
		}
	}

	GridAnchorGenerator* obj = new GridAnchorGenerator(boxParams.data(), numLayers);
	obj->setPluginNamespace(mNamespace.c_str());
	return obj;
}

IPluginV2Ext* mGridAnchorPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
	std::cout << "deserializePlugin" << std::endl;
	// This object will be deleted when the network is destroyed, which will
	// call GridAnchor::destroy()
	GridAnchorGenerator* obj = new GridAnchorGenerator(serialData, serialLength);
	obj->setPluginNamespace(mNamespace.c_str());
	return obj;
}
