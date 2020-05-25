#include <vector>
#include "kernel.h"

template <typename T_BBOX, unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta)
__global__ void decodeBBoxes_kernelPP(
	const int nthreads,
	const CodeTypeSSD code_type,
	const bool variance_encoded_in_target,
	const int num_priors,
	const bool share_location,
	const int num_loc_classes,
	const int background_label_id,
	const bool clip_bbox,
	const T_BBOX* loc_data,
	const T_BBOX* prior_data,
	T_BBOX* bbox_data)
{
	for (int index = blockIdx.x * nthds_per_cta + threadIdx.x;
		index < nthreads;
		index += nthds_per_cta * gridDim.x)
	{
		// Particular points coordinate index {0, 1, 2, 3, 4}
		const int i = index % 10;
		// Particular points set class index
		const int c = (index / 10) % num_loc_classes;
		// Particular points set id corresponding to the particular points
		const int d = (index / 10 / num_loc_classes) % num_priors;
		// If Particular points set was not shared among all the classes and the Particular points set is corresponding to the background class
		if (!share_location && c == background_label_id)
		{
			// Ignore background class if not share_location.
			return;
		}
		// Index to the right anchor box corresponding to the current Particular points 
		const int pi = d * 4;
		// Index to the right variances corresponding to the current Particular points	 
		const int vi = pi + num_priors * 4;

		// Encoding method: CodeTypeSSD::CORNER
		//if (code_type == PriorBoxParameter_CodeType_CORNER){
		if (code_type == CodeTypeSSD::CORNER)
		{
			// Do not want to use variances to adjust the bounding box decoding
			if (variance_encoded_in_target)
			{
				// variance is encoded in target, we simply need to add the offset
				// predictions.
				// prior_data[pi + i]: prior box coordinates corresponding to the current bounding box coordinate
				bbox_data[index] = prior_data[pi + i] + loc_data[index];
			}
			else
			{
				// variance is encoded in bbox, we need to scale the offset accordingly.
				// prior_data[vi + i]: variance corresponding to the current bounding box coordinate
				bbox_data[index] = prior_data[pi + i] + loc_data[index] * prior_data[vi + i];
			}
			//} else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
		}
		// Encoding method: CodeTypeSSD::CENTER_SIZE
		else if (code_type == CodeTypeSSD::CENTER_SIZE)
		{
			// Get prior box coordinates
			const T_BBOX p_xmin = prior_data[pi];
			const T_BBOX p_ymin = prior_data[pi + 1];
			const T_BBOX p_xmax = prior_data[pi + 2];
			const T_BBOX p_ymax = prior_data[pi + 3];

			// Calculate prior box center, height, and width
			/*const T_BBOX prior_width = p_xmax - p_xmin;
			const T_BBOX prior_height = p_ymax - p_ymin;
			const T_BBOX prior_center_x = (p_xmin + p_xmax) / 2.;
			const T_BBOX prior_center_y = (p_ymin + p_ymax) / 2.;*/
			//printf("index = %d, index - i = %d \n", index, index - i);
			// Get the current bounding box coordinates
			const T_BBOX loc_x1 = loc_data[index - i];
			const T_BBOX loc_y1 = loc_data[index - i + 1];
			const T_BBOX loc_x2 = loc_data[index - i + 2];
			const T_BBOX loc_y2 = loc_data[index - i + 3];
			const T_BBOX loc_x3 = loc_data[index - i + 4];
			const T_BBOX loc_y3 = loc_data[index - i + 5];
			const T_BBOX loc_x4 = loc_data[index - i + 6];
			const T_BBOX loc_y4 = loc_data[index - i + 7];
			const T_BBOX loc_x5 = loc_data[index - i + 8];
			const T_BBOX loc_y5 = loc_data[index - i + 9];
			// Declare decoded bounding box coordinates
			T_BBOX decode_bbox_center_x1, decode_bbox_center_x2, decode_bbox_center_x3, decode_bbox_center_x4, decode_bbox_center_x5;
			T_BBOX decode_bbox_center_y1, decode_bbox_center_y2, decode_bbox_center_y3, decode_bbox_center_y4, decode_bbox_center_y5;
			// Do not want to use variances to adjust the bounding box decoding
			//if (variance_encoded_in_target)
			//{
			//	// variance is encoded in target, we simply need to retore the offset
			//	// predictions.
			//	//decode_bbox_center_x1 = xmin * prior_width + prior_center_x;
			//	//decode_bbox_center_y1 = ymin * prior_height + prior_center_y;
			//	//decode_bbox_width = exp(xmax) * prior_width;
			//	//decode_bbox_height = exp(ymax) * prior_height;

			//}
			//else
			//{
				// variance is encoded in bbox, we need to scale the offset accordingly.
				//decode_bbox_center_x = prior_data[vi] * xmin * prior_width + prior_center_x;
				//decode_bbox_center_y = prior_data[vi + 1] * ymin * prior_height + prior_center_y;
				//decode_bbox_width = exp(prior_data[vi + 2] * xmax) * prior_width;
				//decode_bbox_height = exp(prior_data[vi + 3] * ymax) * prior_height;
				decode_bbox_center_x1 = p_xmin + loc_x1 * 0.1 * p_xmax;
				decode_bbox_center_y1 = p_ymin + loc_y1 * 0.1 * p_ymax;
				decode_bbox_center_x2 = p_xmin + loc_x2 * 0.1 * p_xmax;
				decode_bbox_center_y2 = p_ymin + loc_y2 * 0.1 * p_ymax;
				decode_bbox_center_x3 = p_xmin + loc_x3 * 0.1 * p_xmax;
				decode_bbox_center_y3 = p_ymin + loc_y3 * 0.1 * p_ymax;
				decode_bbox_center_x4 = p_xmin + loc_x4 * 0.1 * p_xmax;
				decode_bbox_center_y4 = p_ymin + loc_y4 * 0.1 * p_ymax;
				decode_bbox_center_x5 = p_xmin + loc_x5 * 0.1 * p_xmax;
				decode_bbox_center_y5 = p_ymin + loc_y5 * 0.1 * p_ymax;
			//}

			// Use [x_topleft, y_topleft, x_bottomright, y_bottomright] as coordinates for final decoded bounding box output
			
			switch (i)
			{
			case 0:
				bbox_data[index] = decode_bbox_center_x1;
				break;
			case 1:
				bbox_data[index] = decode_bbox_center_y1;
				break;
			case 2:
				bbox_data[index] = decode_bbox_center_x2; //decode_bbox_center_x + decode_bbox_width / 2.;
				break;
			case 3:
				bbox_data[index] = decode_bbox_center_y2; //decode_bbox_center_y + decode_bbox_height / 2.;
				break;
			case 4:
				bbox_data[index] = decode_bbox_center_x3; //decode_bbox_center_y + decode_bbox_height / 2.;
				break;
			case 5:
				bbox_data[index] = decode_bbox_center_y3;
				break;
			case 6:
				bbox_data[index] = decode_bbox_center_x4;
				break;
			case 7:
				bbox_data[index] = decode_bbox_center_y4; //decode_bbox_center_x + decode_bbox_width / 2.;
				break;
			case 8:
				bbox_data[index] = decode_bbox_center_x5; //decode_bbox_center_y + decode_bbox_height / 2.;
				break;
			case 9:
				bbox_data[index] = decode_bbox_center_y5; //decode_bbox_center_y + decode_bbox_height / 2.;
				break;
			}
			//} else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
		}
		// Encoding method: CodeTypeSSD::CORNER_SIZE
		else if (code_type == CodeTypeSSD::CORNER_SIZE)
		{
			// Get prior box coordinates
			const T_BBOX p_xmin = prior_data[pi];
			const T_BBOX p_ymin = prior_data[pi + 1];
			const T_BBOX p_xmax = prior_data[pi + 2];
			const T_BBOX p_ymax = prior_data[pi + 3];
			// Get prior box width and height
			const T_BBOX prior_width = p_xmax - p_xmin;
			const T_BBOX prior_height = p_ymax - p_ymin;
			T_BBOX p_size;
			if (i == 0 || i == 2)
			{
				p_size = prior_width;
			}
			else
			{
				p_size = prior_height;
			}
			// Do not want to use variances to adjust the bounding box decoding
			if (variance_encoded_in_target)
			{
				// variance is encoded in target, we simply need to add the offset
				// predictions.
				bbox_data[index] = prior_data[pi + i] + loc_data[index] * p_size;
			}
			else
			{
				// variance is encoded in bbox, we need to scale the offset accordingly.
				bbox_data[index] = prior_data[pi + i] + loc_data[index] * prior_data[vi + i] * p_size;
			}
		}
		// Exactly the same to CodeTypeSSD::CENTER_SIZE with using variance to adjust the bounding box decoding 
		else if (code_type == CodeTypeSSD::TF_CENTER)
		{
			const T_BBOX pXmin = prior_data[pi];
			const T_BBOX pYmin = prior_data[pi + 1];
			const T_BBOX pXmax = prior_data[pi + 2];
			const T_BBOX pYmax = prior_data[pi + 3];
			const T_BBOX priorWidth = pXmax - pXmin;
			const T_BBOX priorHeight = pYmax - pYmin;
			const T_BBOX priorCenterX = (pXmin + pXmax) / 2.;
			const T_BBOX priorCenterY = (pYmin + pYmax) / 2.;

			const T_BBOX ymin = loc_data[index - i];
			const T_BBOX xmin = loc_data[index - i + 1];
			const T_BBOX ymax = loc_data[index - i + 2];
			const T_BBOX xmax = loc_data[index - i + 3];

			T_BBOX bboxCenterX, bboxCenterY;
			T_BBOX bboxWidth, bboxHeight;

			bboxCenterX = prior_data[vi] * xmin * priorWidth + priorCenterX;
			bboxCenterY = prior_data[vi + 1] * ymin * priorHeight + priorCenterY;
			bboxWidth = exp(prior_data[vi + 2] * xmax) * priorWidth;
			bboxHeight = exp(prior_data[vi + 3] * ymax) * priorHeight;

			switch (i)
			{
			case 0:
				bbox_data[index] = bboxCenterX - bboxWidth / 2.;
				break;
			case 1:
				bbox_data[index] = bboxCenterY - bboxHeight / 2.;
				break;
			case 2:
				bbox_data[index] = bboxCenterX + bboxWidth / 2.;
				break;
			case 3:
				bbox_data[index] = bboxCenterY + bboxHeight / 2.;
				break;
			}
		}
		else
		{
			// Unknown code type.
			assert("Unknown Box decode code type");
		}
		// Clip bounding box or not
		if (clip_bbox)
		{
			bbox_data[index] = max(min(bbox_data[index], T_BBOX(1.)), T_BBOX(0.));
		}
	}
	//for (int index = blockIdx.x * nthds_per_cta + threadIdx.x;
	//	index < nthreads;
	//	index += nthds_per_cta * gridDim.x)
	//{
	//	printf("index %d: %f \n", index, bbox_data[index]);
	//}
}

template <typename T_BBOX>
pluginStatus_t decodeBBoxes_gpuPP(
	cudaStream_t stream,
	const int nthreads,
	const CodeTypeSSD code_type,
	const bool variance_encoded_in_target,
	const int num_priors,
	const bool share_location,
	const int num_loc_classes,
	const int background_label_id,
	const bool clip_bbox,
	const void* loc_data,
	const void* prior_data,
	void* bbox_data)
{
	const int BS = 512;
	const int GS = (nthreads + BS - 1) / BS;
	decodeBBoxes_kernelPP<T_BBOX, BS> << <GS, BS, 0, stream >> > (nthreads, code_type, variance_encoded_in_target,
		num_priors, share_location, num_loc_classes,
		background_label_id, clip_bbox,
		(const T_BBOX*)loc_data, (const T_BBOX*)prior_data,
		(T_BBOX*)bbox_data);
	CSC(cudaGetLastError(), STATUS_FAILURE);
	return STATUS_SUCCESS;
}

// decodeBBoxes LAUNCH CONFIG
typedef pluginStatus_t(*dbbFunc)(cudaStream_t,
	const int,
	const CodeTypeSSD,
	const bool,
	const int,
	const bool,
	const int,
	const int,
	const bool,
	const void*,
	const void*,
	void*);

struct dbbLaunchConfig
{
	DataType t_bbox;
	dbbFunc function;

	dbbLaunchConfig(DataType t_bbox)
		: t_bbox(t_bbox)
	{
	}
	dbbLaunchConfig(DataType t_bbox, dbbFunc function)
		: t_bbox(t_bbox)
		, function(function)
	{
	}
	bool operator==(const dbbLaunchConfig& other)
	{
		return t_bbox == other.t_bbox;
	}
};

static std::vector<dbbLaunchConfig> dbbFuncVec;

bool decodeBBoxesInitPP()
{
	dbbFuncVec.push_back(dbbLaunchConfig(DataType::kFLOAT, decodeBBoxes_gpuPP<float>));
	return true;
}

static bool initialized = decodeBBoxesInitPP();

pluginStatus_t decodeBBoxesPP(
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
	void* bbox_data)
{
	dbbLaunchConfig lc = dbbLaunchConfig(DT_BBOX);
	for (unsigned i = 0; i < dbbFuncVec.size(); ++i)
	{
		if (lc == dbbFuncVec[i])
		{
			DEBUG_PRINTF("decodeBBox kernel %d\n", i);
			return dbbFuncVec[i].function(stream,
				nthreads,
				code_type,
				variance_encoded_in_target,
				num_priors,
				share_location,
				num_loc_classes,
				background_label_id,
				clip_bbox,
				loc_data,
				prior_data,
				bbox_data);
		}
	}
	return STATUS_BAD_PARAM;
}