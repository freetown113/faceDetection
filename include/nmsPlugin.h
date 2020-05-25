#pragma once
//#include "kernel.h"
#include "plugin.h"
#include <string>
#include <vector>
#include <algorithm>

using namespace nvinfer1;
using namespace nvinfer1::plugin;

pluginStatus_t detectionInference(
	cudaStream_t stream,
	const int N,
	const int C0,
	const int C1,
	const int C2,
	const bool shareLocation,
	const bool varianceEncodedInTarget,
	const int backgroundLabelId,
	const int numPredsPerClass,
	const int numClasses,
	const int topK,
	const int keepTopK,
	const float confidenceThreshold,
	const float nmsThreshold,
	const CodeTypeSSD codeType,
	const DataType DT_BBOX,
	const void* locDataPoints,
	const int offset,
	const void* locDataRects,
	const void* priorData,
	const DataType DT_SCORE,
	const void* confData,
	void* keepCount,
	void* topDetectionsPoints,
	void* topDetectionsRects,
	void* workspace,
	bool isNormalized,
	bool confSigmoid);

//____________________________________________________________________________
size_t detectionForwardBBoxDataSize(int N, int C1, DataType DT_BBOX);

size_t detectionForwardBBoxPermuteSize(bool shareLocation, int N, int C1, DataType DT_BBOX);

size_t detectionForwardPreNMSSize(int N, int C2);

size_t detectionForwardPostNMSSize(int N, int numClasses, int topK);

//____________________________________________________________________________

size_t sortScoresPerClassWorkspaceSize(
	const int num,
	const int num_classes,
	const int num_preds_per_class,
	const DataType DT_CONF);

size_t sortScoresPerImageWorkspaceSize(
	const int num_images,
	const int num_items_per_image,
	const DataType DT_SCORE);

size_t calculateTotalWorkspaceSize(size_t* workspaces, int count);

size_t detectionInferenceWorkspaceSize(bool shareLocation, int N, int C1, int C2, int numClasses, int numPredsPerClass,
	int topK, DataType DT_BBOX, DataType DT_SCORE);

struct DetectionOutputParametersPP
{
public:
	bool shareLocation, varianceEncodedInTarget;
	int backgroundLabelId, numClasses, topK, keepTopK;
	float confidenceThreshold, nmsThreshold;
	CodeTypeSSD codeType;
	int inputOrder[4];
	bool confSigmoid;
	bool isNormalized;
};

//using namespace nvinfer1::plugin;
namespace nvinfer1
{
	namespace plugin
	{

		class DetectionOutput : public IPluginV2Ext
		{
		public:
			DetectionOutput(DetectionOutputParametersPP param);

			DetectionOutput(DetectionOutputParametersPP param, int C0, int C1, int C2, int numPriors);

			DetectionOutput(const void* data, size_t length);

			~DetectionOutput() override = default;

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
			DetectionOutputParametersPP param;
			int C0, C1, C2, numPriors;
			const char* mPluginNamespace;
		};

		class mNMSPluginCreator : public BaseCreator
		{
		public:
			mNMSPluginCreator();

			~mNMSPluginCreator() override = default;

			const char* getPluginName() const override;

			const char* getPluginVersion() const override;

			const PluginFieldCollection* getFieldNames() override;

			IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

			IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

		private:
			static PluginFieldCollection mFC;

			// Parameters for DetectionOutput
			DetectionOutputParametersPP params;
			static std::vector<PluginField> mPluginAttributes;
		};
	} // namespace plugin
} // namespace nvinfer1
//REGISTER_TENSORRT_PLUGIN(nvinfer1::plugin::mNMSPluginCreator);
