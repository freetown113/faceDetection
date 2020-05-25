#pragma once
#include"gpuprocess.h"
#include"cuda_includes.h"

namespace gpuprocess
{
	class Logger : public nvinfer1::ILogger
	{
	public:
		Logger(Severity severity = Severity::kWARNING)
			: reportableSeverity(severity)
		{}

		void log(Severity severity, const char* msg) override
		{
			// suppress messages with severity enum value greater than the reportable
			if (severity > reportableSeverity)
				return;

			switch (severity)
			{
			case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: ";
				break;
			case Severity::kERROR: std::cerr << "ERROR: ";
				break;
			case Severity::kWARNING: std::cerr << "WARNING: ";
				break;
			case Severity::kINFO: std::cerr << "INFO: ";
				break;
			default:
				std::cerr << "UNKNOWN: ";
				break;
			}
			std::cerr << msg << std::endl;
		}
		Severity reportableSeverity;
	};

	class TRuntime : public gpuprocess::ITRTRuntime
	{
	public:
		TRuntime(NvInferPtr<nvinfer1::ICudaEngine> &&eng, int batch);
		virtual ~TRuntime() = default;
		bool saveEngine(const char *path) override;
		void inference(std::vector<void*>& outputsBuffers) override;
		inline void print_version()
		{
			std::cout << "  TensorRT version: "
				<< NV_TENSORRT_MAJOR << "."
				<< NV_TENSORRT_MINOR << "."
				<< NV_TENSORRT_PATCH << "."
				<< NV_TENSORRT_BUILD << std::endl;
		}
	private:
		NvInferPtr<nvinfer1::ICudaEngine> _engine;
		NvInferPtr<nvinfer1::IExecutionContext> _context;
		int batch_size;
	};
}