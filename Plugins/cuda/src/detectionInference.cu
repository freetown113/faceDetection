//#include "kernel.h"
#include "bboxUtils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "plugin.h"
#include<NvInferPlugin.h>
#include<NvInferPluginUtils.h>
#include<NvInfer.h>
#include<NvInferRuntime.h>

using namespace nvinfer1;
using namespace nvinfer1::plugin;

pluginStatus_t sortScoresPerImage(
	cudaStream_t stream,
	const int num_images,
	const int num_items_per_image,
	const DataType DT_SCORE,
	void* unsorted_scores,
	void* unsorted_bbox_indices,
	void* sorted_scores,
	void* sorted_bbox_indices,
	void* workspace);

pluginStatus_t sortScoresPerClass(
	cudaStream_t stream,
	const int num,
	const int num_classes,
	const int num_preds_per_class,
	const int background_label_id,
	const float confidence_threshold,
	const DataType DT_SCORE,
	void* conf_scores_gpu,
	void* index_array_gpu,
	void* workspace);

pluginStatus_t decodeBBoxes(
	cudaStream_t stream,
	const int nthreads,
	const CodeTypeSSD code_type,
	const bool variance_encoded_in_target,
	const int num_priors,
	const bool share_location,
	const int num_loc_classes,
	const int background_label_id,
	const bool clip_bbox,
	const DataType DT_BBOX,
	const void* loc_data,
	const void* prior_data,
	void* bbox_data);

pluginStatus_t permuteData(cudaStream_t stream,
	const int nthreads,
	const int num_classes,
	const int num_data,
	const int num_dim,
	const DataType DT_DATA,
	bool confSigmoid,
	const void* data,
	void* new_data);

pluginStatus_t gatherTopDetections(
	cudaStream_t stream,
	const bool shareLocation,
	const int numImages,
	const int numPredsPerClass,
	const int numClasses,
	const int topK,
	const int keepTopK,
	const DataType DT_BBOX,
	const DataType DT_SCORE,
	const void* indices,
	const void* scores,
	const void* bboxData,
	void* keepCount,
	void* topDetections);

pluginStatus_t allClassNMS_m(cudaStream_t stream,
	const int num,
	const int num_classes,
	const int num_preds_per_class,
	const int top_k,
	const float nms_threshold,
	const bool share_location,
	const bool isNormalized,
	const DataType DT_SCORE,
	const DataType DT_BBOX,
	void* bbox_data,
	void* beforeNMS_scores,
	void* beforeNMS_index_array,
	void* afterNMS_scores,
	void* afterNMS_index_array,
	bool flipXY);

size_t detectionForwardBBoxDataSize(int N, int C1, DataType DT_BBOX)
{
	if (DT_BBOX == DataType::kFLOAT)
	{
		return N * C1 * sizeof(float);
	}

	printf("Only FP32 type bounding boxes are supported.\n");
	return (size_t)-1;
}

size_t detectionForwardBBoxPermuteSize(bool shareLocation, int N, int C1, DataType DT_BBOX)
{
	if (DT_BBOX == DataType::kFLOAT)
	{
		return shareLocation ? 0 : N * C1 * sizeof(float);
	}
	printf("Only FP32 type bounding boxes are supported.\n");
	return (size_t)-1;
}

size_t detectionForwardPreNMSSize(int N, int C2)
{
	ASSERT(sizeof(float) == sizeof(int));
	return N * C2 * sizeof(float);
}

size_t detectionForwardPostNMSSize(int N, int numClasses, int topK)
{
	ASSERT(sizeof(float) == sizeof(int));
	return N * numClasses * topK * sizeof(float);
}


pluginStatus_t detectionInference(
	cudaStream_t stream,
	const int N,
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
	const void* locData,
	const void* priorData,
	const DataType DT_SCORE,
	const void* confData,
	void* keepCount,
	void* topDetections,
	void* workspace,
	bool isNormalized,
	bool confSigmoid)
{
	// Batch size * number bbox per sample * 4 = total number of bounding boxes * 4
	const int locCount = N * C1;
	// Do not clip the bounding box that goes outside the image
	const bool clipBBox = false;
	/*
	 * shareLocation
	 * Bounding box are shared among all classes, i.e., a bounding box could be classified as any candidate class.
	 * Otherwise
	 * Bounding box are designed for specific classes, i.e., a bounding box could be classified as one certain class or not (binary classification).
	 */
	const int numLocClasses = shareLocation ? 1 : numClasses;

	size_t bboxDataSize = detectionForwardBBoxDataSize(N, C1, DataType::kFLOAT);  //for pPoints 5250 
	void* bboxDataRaw = workspace;

	pluginStatus_t status = decodeBBoxes(stream,
		locCount,
		codeType,
		varianceEncodedInTarget,
		numPredsPerClass,
		shareLocation,
		numLocClasses,
		backgroundLabelId,
		clipBBox,
		DataType::kFLOAT,
		locData,
		priorData,
		bboxDataRaw);

	ASSERT_FAILURE(status == STATUS_SUCCESS);

	/*
	 * bboxDataRaw format:
	 * [batch size, numPriors (per sample), numLocClasses, 4]
	 */
	 // float for now
	void* bboxData;
	size_t bboxPermuteSize = detectionForwardBBoxPermuteSize(shareLocation, N, C1, DataType::kFLOAT); // for pPoints 0
	void* bboxPermute = nextWorkspacePtr((int8_t*)bboxDataRaw, bboxDataSize);

	/*
	 * After permutation, bboxData format:
	 * [batch_size, numLocClasses, numPriors (per sample) (numPredsPerClass), 4]
	 * This is equivalent to swapping axis
	 */
	if (!shareLocation)
	{
		status = permuteData(stream,
			locCount,
			numLocClasses,
			numPredsPerClass,
			4,
			DataType::kFLOAT,
			false,
			bboxDataRaw,
			bboxPermute);
		ASSERT_FAILURE(status == STATUS_SUCCESS);
		bboxData = bboxPermute;
	}
	/*
	 * If shareLocation, numLocClasses = 1
	 * No need to permute data on linear memory
	 */
	else
	{
		bboxData = bboxDataRaw;
	}
	/*
	 * Conf data format
	 * [batch size, numPriors * param.numClasses, 1, 1]
	 */
	const int numScores = N * C2;
	size_t scoresSize = detectionForwardPreNMSSize(N, C2); //for pPoints 4200
	void* scores = nextWorkspacePtr((int8_t*)bboxPermute, bboxPermuteSize);
	// need a conf_scores
	/*
	 * After permutation, bboxData format:
	 * [batch_size, numClasses, numPredsPerClass, 1]
	 */
	status = permuteData(stream,
		numScores,
		numClasses,
		numPredsPerClass,
		1,
		DataType::kFLOAT,
		confSigmoid,
		confData,
		scores);
	ASSERT_FAILURE(status == STATUS_SUCCESS);

	size_t indicesSize = detectionForwardPreNMSSize(N, C2); //for pPoints 4200
	void* indices = nextWorkspacePtr((int8_t*)scores, scoresSize);

	size_t postNMSScoresSize = detectionForwardPostNMSSize(N, numClasses, topK); //for pPoints 1600
	size_t postNMSIndicesSize = detectionForwardPostNMSSize(N, numClasses, topK); //for pPoints 1600
	void* postNMSScores = nextWorkspacePtr((int8_t*)indices, indicesSize);
	void* postNMSIndices = nextWorkspacePtr((int8_t*)postNMSScores, postNMSScoresSize);

	//size_t sortingWorkspaceSize = sortScoresPerClassWorkspaceSize(N, numClasses, numPredsPerClass, FLOAT32);
	void* sortingWorkspace = nextWorkspacePtr((int8_t*)postNMSIndices, postNMSIndicesSize);
	// Sort the scores so that the following NMS could be applied.
	status = sortScoresPerClass(stream,
		N,
		numClasses,
		numPredsPerClass,
		backgroundLabelId,
		confidenceThreshold,
		DataType::kFLOAT,
		scores,
		indices,
		sortingWorkspace);
	ASSERT_FAILURE(status == STATUS_SUCCESS);

	// NMS
	status = allClassNMS_m(stream,
		N,
		numClasses,
		numPredsPerClass,
		topK,
		nmsThreshold,
		shareLocation,
		isNormalized,
		DataType::kFLOAT,
		DataType::kFLOAT,
		bboxData,
		scores,
		indices,
		postNMSScores,
		postNMSIndices,
		false);
	ASSERT_FAILURE(status == STATUS_SUCCESS);

	// Sort the bounding boxes after NMS using scores
	status = sortScoresPerImage(stream,
		N,
		numClasses * topK,
		DataType::kFLOAT,
		postNMSScores,
		postNMSIndices,
		scores,
		indices,
		sortingWorkspace);
	ASSERT_FAILURE(status == STATUS_SUCCESS);

	// Gather data from the sorted bounding boxes after NMS
	status = gatherTopDetections(stream,
		shareLocation,
		N,
		numPredsPerClass,
		numClasses,
		topK,
		keepTopK,
		DataType::kFLOAT,
		DataType::kFLOAT,
		indices,
		scores,
		bboxData,
		keepCount,
		topDetections);
	ASSERT_FAILURE(status == STATUS_SUCCESS);

	return STATUS_SUCCESS;
}
