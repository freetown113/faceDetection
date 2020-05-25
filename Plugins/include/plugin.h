#pragma once
#include "NvInferPlugin.h"
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <string>

// Enumerator for status
typedef enum
{
	STATUS_SUCCESS = 0,
	STATUS_FAILURE = 1,
	STATUS_BAD_PARAM = 2,
	STATUS_NOT_SUPPORTED = 3,
	STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

namespace nvinfer1
{
	namespace plugin
	{

		class BaseCreator : public IPluginCreator
		{
		public:
			void setPluginNamespace(const char* libNamespace) override
			{
				mNamespace = libNamespace;
			}

			const char* getPluginNamespace() const override
			{
				return mNamespace.c_str();
			}

		protected:
			std::string mNamespace;
		};

		// Write values into buffer
		template <typename T>
		void write(char*& buffer, const T& val)
		{
			*reinterpret_cast<T*>(buffer) = val;
			buffer += sizeof(T);
		}

		// Read values from buffer
		template <typename T>
		T read(const char*& buffer)
		{
			T val = *reinterpret_cast<const T*>(buffer);
			buffer += sizeof(T);
			return val;
		}

	} // namespace plugin
} // namespace nvinfer1

pluginStatus_t priorBoxInference(cudaStream_t stream,
	const nvinfer1::plugin::PriorBoxParameters param,
	const int H,
	const int W,
	const int numPriors,
	const int numAspectRatios,
	const void* minSize,
	const void* maxSize,
	const void* aspectRatios,
	void* outputData);



#ifndef DEBUG

#define ASSERT(assertion)                                                                                              \
    {                                                                                                                  \
        if (!(assertion))                                                                                              \
        {                                                                                                              \
            std::cerr << "#assertion" << __FILE__ << "," << __LINE__ << std::endl;                                     \
            abort();                                                                                                   \
        }                                                                                                              \
    }

#define CUASSERT(status_)                                                                                              \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != cudaSuccess)                                                                                         \
        {                                                                                                              \
            std::cerr << __FILE__ << ", " << __LINE__ << ", " << s_ << ", " << cudaGetErrorString(s_) << std::endl;    \
        }                                                                                                              \
    }
#define CUBLASASSERT(status_)                                                                                          \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != CUBLAS_STATUS_SUCCESS)                                                                               \
        {                                                                                                              \
            std::cerr << __FILE__ << ", " << __LINE__ << ", " << s_ << std::endl;                                      \
        }                                                                                                              \
    }
#define CUERRORMSG(status_)                                                                                            \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != 0)                                                                                                   \
            std::cerr << __FILE__ << ", " << __LINE__ << ", " << s_ << std::endl;                                      \
    }
#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        if (status != 0)                                                                                               \
            abort();                                                                                                   \
    } while (0)

#define ASSERT_PARAM(exp)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
            return STATUS_BAD_PARAM;                                                                                   \
    } while (0)

#define ASSERT_FAILURE(exp)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
            return STATUS_FAILURE;                                                                                     \
    } while (0)

#define CSC(call, err)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = call;                                                                                 \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            return err;                                                                                                \
        }                                                                                                              \
    } while (0)

#define DEBUG_PRINTF(...)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)

#else

#define ASSERT_PARAM(exp)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
        {                                                                                                              \
            fprintf(stderr, "Bad param - " #exp ", %s:%d\n", __FILE__, __LINE__);                                      \
            return STATUS_BAD_PARAM;                                                                                   \
        }                                                                                                              \
    } while (0)

#define ASSERT_FAILURE(exp)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
        {                                                                                                              \
            fprintf(stderr, "Failure - " #exp ", %s:%d\n", __FILE__, __LINE__);                                        \
            return STATUS_FAILURE;                                                                                     \
        }                                                                                                              \
    } while (0)

#define CSC(call, err)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = call;                                                                                 \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            printf("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus));                        \
            return err;                                                                                                \
        }                                                                                                              \
    } while (0)

#define CHECK(status)                                                                                                  \
    {                                                                                                                  \
        if (status != 0)                                                                                               \
        {                                                                                                              \
            DEBUG_PRINTF("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(status));                      \
            abort();                                                                                                   \
        }                                                                                                              \
    }

#define DEBUG_PRINTF(...)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        printf(__VA_ARGS__);                                                                                           \
    } while (0)

#endif