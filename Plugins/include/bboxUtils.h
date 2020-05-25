#pragma once
#include "plugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;

template <typename T>
struct Bbox
{
	T xmin, ymin, xmax, ymax;
	Bbox(T xmin, T ymin, T xmax, T ymax)
		: xmin(xmin)
		, ymin(ymin)
		, xmax(xmax)
		, ymax(ymax)
	{
	}
	Bbox() = default;
};

template <typename T>
struct BboxInfo
{
	T conf_score;
	int label;
	int bbox_idx;
	bool kept;
	BboxInfo(T conf_score, int label, int bbox_idx, bool kept)
		: conf_score(conf_score)
		, label(label)
		, bbox_idx(bbox_idx)
		, kept(kept)
	{
	}
	BboxInfo() = default;
};

template <typename TFloat>
bool operator<(const Bbox<TFloat>& lhs, const Bbox<TFloat>& rhs)
{
	return lhs.x1 < rhs.x1;
}

template <typename TFloat>
bool operator==(const Bbox<TFloat>& lhs, const Bbox<TFloat>& rhs)
{
	return lhs.x1 == rhs.x1 && lhs.y1 == rhs.y1 && lhs.x2 == rhs.x2 && lhs.y2 == rhs.y2;
}
// }}}

int8_t* alignPtr(int8_t* ptr, uintptr_t to);

int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize);

size_t dataTypeSize(DataType dtype);

void setUniformOffsets(cudaStream_t stream, int num_segments, int offset, int* d_offsets);