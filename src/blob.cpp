#include"gpuprocess.h"
#include"cuda_includes.h"

using namespace gpuprocess;

void swapp(void* &a, void* &b)
{
	void* t = a;
	a = b;
	b = t;
}

void gpuprocess::swap(Blob& one, Blob& another)
{
	using std::swap;
	swap(one.height, another.height);
	swap(one.width, another.width);
	swap(one.channels, another.channels);
	swap(one.element_size, another.element_size);
	swapp(one.data, another.data);
}

Blob::Blob(int h, int w, int ch, int el_size)
	:height(h), width(w), channels(ch), element_size(el_size)
{
	CUDA_SAFE_CALL(cudaMalloc(&this->data, this->height * this->width * this->channels * this->element_size));
	ownership = true;
}

Blob::Blob(void* buffer, int h, int w, int ch, int el_size)
	: height(h), width(w), channels(ch), element_size(el_size)
{
	ownership = true;
	data = buffer;
}

Blob::~Blob()
{
	if (ownership && data != nullptr)
		cudaFree(data);
}

Blob::Blob(Blob const& copy)
	:width(copy.width), height(copy.height), channels(copy.channels), element_size(copy.element_size), data(nullptr)
{
	std::cout << "Object copied!" << std::endl;
	ownership = true;
	cudaMalloc(&data, width * height * channels * element_size);
}

//Blob::Blob(Blob&& copy)
//	:width(copy.width), height(copy.height), channels(copy.channels), element_size(copy.element_size), data(copy.data),
//	ownership(copy.ownership)
//{
//	std::cout << "Object moved!" << std::endl;	
//	copy.data = nullptr;
//	copy.ownership = false;
//}

Blob::Blob(Blob&& copy) :
	Blob()
{
	std::cout << "Object moved!" << std::endl;
	swap(*this, copy);
}

//Blob& Blob::operator=(const Blob& copy)
//{
//	if (this != &copy)
//	{
//		std::cout << "Object assigned!" << std::endl;
//		if (copy.ownership == true)
//			ownership = true;
//		else
//			ownership = false;
//		width = copy.width;
//		height = copy.height;
//		channels = copy.channels;
//		element_size = copy.element_size;
//		cudaFree(data);
//		cudaMalloc(&data, width * height * channels * element_size);
//	}
//	return *this;
//}

Blob& Blob::operator=(Blob copy)
{
	std::cout << "Object assigned!" << std::endl;
	swap(*this, copy);
	return *this;
}

//gpuprocess::Blob gpuprocess::allocate_memory(int height, int width, int channels, int element_size)
//{
//	//Blob* newBuffer = new Blob(width, height, channels, element_size);
//	Blob newBuffer(width, height, channels, element_size);
//	CUDA_SAFE_CALL(cudaMalloc(newBuffer.getDataPtrAddr(), newBuffer.getBufferSize() * newBuffer.getElemSize()));
//	return newBuffer;
//}