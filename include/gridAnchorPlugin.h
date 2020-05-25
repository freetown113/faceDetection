#pragma once
#include "cudnn.h"
#include "plugin.h"
#include <cublas_v2.h>
#include <string>
#include <vector>
#include <NvOnnxParser.h>

pluginStatus_t anchorGridInference(
	cudaStream_t stream,
	const nvinfer1::plugin::GridAnchorParameters param,
	const int numAspectRatios,
	const void* widths,
	const void* heights,
	void* outputData);


namespace nvinfer1
{
	namespace plugin
	{
		class GridAnchorGenerator : public IPluginV2Ext
		{
		public:
			GridAnchorGenerator(const GridAnchorParameters* param, int numLayers);

			GridAnchorGenerator(const void* data, size_t length);

			~GridAnchorGenerator() override;

			int getNbOutputs() const override;

			Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

			int initialize() override;

			void terminate() override;

			size_t getWorkspaceSize(int maxBatchSize) const override;

			int enqueue(
				int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

			size_t getSerializationSize() const override;

			void serialize(void* buffer) const override;

			bool supportsFormat(DataType type, PluginFormat format) const override;

			const char* getPluginType() const override;

			const char* getPluginVersion() const override;

			void destroy() override;

			IPluginV2Ext* clone() const override;

			void setPluginNamespace(const char* pluginNamespace) override;

			const char* getPluginNamespace() const override;

			DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

			bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

			bool canBroadcastInputAcrossBatch(int inputIndex) const override;

			void attachToContext(
				cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

			void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
				const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
				const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;

			void detachFromContext() override;

		private:
			Weights copyToDevice(const void* hostData, size_t count);

			void serializeFromDevice(char*& hostBuffer, Weights deviceWeights) const;

			Weights deserializeToDevice(const char*& hostBuffer, size_t count);

			int mNumLayers;
			std::vector<GridAnchorParameters> mParam;
			int* mNumPriors;
			Weights *mDeviceWidths, *mDeviceHeights;
			const char* mPluginNamespace;
		};

		class mGridAnchorPluginCreator : public BaseCreator
		{
		public:
			mGridAnchorPluginCreator();

			~mGridAnchorPluginCreator() override = default;

			const char* getPluginName() const override;

			const char* getPluginVersion() const override;

			const PluginFieldCollection* getFieldNames() override;

			IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

			IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

		private:
			static PluginFieldCollection mFC;
			static std::vector<PluginField> mPluginAttributes;
		};
	} // namespace plugin
} // namespace nvinfer1
//REGISTER_TENSORRT_PLUGIN(nvinfer1::plugin::mGridAnchorPluginCreator);
