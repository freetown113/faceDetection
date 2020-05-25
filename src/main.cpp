#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <cmath>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <chrono>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
//#include "bioVector.h"
#include "gpuprocess.h"


std::string type2str(int type) 
{
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) 
	{
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
	}
	r += "C";
	r += (chans + '0');
	return r;
}

void norm_tmpl(std::vector<float>& tmplt)
{
	float sumA = 0;
	for (int i = 0; i < tmplt.size(); i++)
		sumA += tmplt[i] * tmplt[i];

	sumA = sqrt(sumA);

	for (int i = 0; i < tmplt.size(); i++)
		tmplt[i] /= sumA;
}

float match_templates(const std::vector<float>& tmplt_a, const std::vector<float>& tmplt_b)
{
	float AB = 0;
	const float* vector_a = tmplt_a.data();
	const float* vector_b = tmplt_b.data();
	for (int i = 0; i < tmplt_a.size(); i++)
	{
		AB += vector_a[i] * vector_b[i];
	}

	return (AB + 1.0f) / 2.0f;
}


int main(int argc, char** argv)
{	
	
	if(argc < 2)
	{
		std::cerr << "invalid arguments quantity, should be 2" << std::endl;
		return 1;
	}
	//Create detector
	gpuprocess::INetBuilder* instance = gpuprocess::getNetBuilder("DETECTION_FACE");
	gpuprocess::ITRTRuntime* inferenceDetector = instance->create(gpuprocess::INetBuilder::LOAD_FROM_MODEL,
		"../ssd_face_detector_shufflenet_v2_640v_with_points_tensorrt_andrey_v.onnx", 1);
	//gpuprocess::ITRTRuntime* inferenceDetector = instance->create(gpuprocess::INetBuilder::LOAD_FROM_ENGINE,
	//	"./engineDet", 1);
	delete instance;
	//Create distinguisher
	//gpuprocess::INetBuilder* instance2 = gpuprocess::getNetBuilder("DISTINGUISH_FACE");
	//gpuprocess::ITRTRuntime* inferenceDistinguish = instance2->create(gpuprocess::INetBuilder::LOAD_FROM_MODEL,
	//	"./model-0000-intermediate3.onnx", 1);
	//delete instance2;	
	
	//Allocate GPU/CPU space for Detector
	gpuprocess::Blob D_imgBuffer(640, 640, 3, sizeof(uint8_t));
	gpuprocess::Blob D_imgBuffer_resized(640, 640, 3, sizeof(uint8_t));
	gpuprocess::Blob D_imgBuffer_32f(640, 640, 3, sizeof(float));
	gpuprocess::Blob D_imgBuffer_transposed(640, 640, 3, sizeof(float));

	float* D_detPoints = new float[1 * 200 * 13];
	gpuprocess::Blob D_detectionPoints(1, 1300, 2, sizeof(float));
	float* D_detRects = new float[1 * 200 * 7];
	gpuprocess::Blob D_detectionRects(1, 700, 2, sizeof(float));

	std::vector<void*> D_inputBuffer({ D_imgBuffer_transposed.getDataPtr(), D_detectionPoints.getDataPtr(), D_detectionRects.getDataPtr() });

	//Allocate GPU/CPU space for Distinguisher
	gpuprocess::Blob P_imgBuffer_resized(112, 112, 3, sizeof(uint8_t));
	gpuprocess::Blob P_imgBuffer_resized_packed(112, 112, 3, sizeof(uint8_t));
	gpuprocess::Blob P_imgBuffer_resized_32(112, 112, 3, sizeof(float));

	//gpuprocess::Blob P_imgBuffer(640, 640, 3, sizeof(uint8_t));
	//gpuprocess::Blob P_imgBuffer_transposed(640, 640, 3, sizeof(uint8_t));
	uint8_t* P_img112 = new uint8_t[112 * 112 * 3];
	float* P_detVector = new float[1 * 512 * 1];
	gpuprocess::Blob P_detectionVector(1, 512, 1, sizeof(float));

	std::vector<void*> P_inputBuffer({ P_imgBuffer_resized_32.getDataPtr(), P_detectionVector.getDataPtr() });

	//~~~~~~~~~~~~~~~~~~~~~~~~
	//std::experimental::filesystem::path source("\\\\lab3030n/Shared/Biometric/Faces/FPI_test_base FULL");
	//std::experimental::filesystem::path source("\\\\lab3030/Shared/ToIvan/RT199_visa");
	std::vector<std::pair<size_t, std::vector<float>>> person_tmplts;
	std::hash<std::string> hash_fn;

	int detect_count(0);	
	int person_count = 0;

	struct dirent* dp = nullptr;
	DIR* fd = nullptr;
	const char* dir = argv[1];	
	if((fd = opendir(argv[1])) == nullptr)
	{
		std::cerr << "Directory " << argv[1] << " doesn't exist" 
		<< " or path is not correct" << std::endl;
		return -1;
	}
	
	//Access to a directory with images
	try
	{
		double all = 0;
		int total = 0;
		//for (auto &&entry : std::experimental::filesystem::recursive_directory_iterator("\\\\lab3030n/Shared/Biometric/Faces/FPI_test_base FULL"))
		//for (auto &&entry : std::experimental::filesystem::recursive_directory_iterator("D:/test/Melanchon/pic"))
		//{
		int id = 1;
		while ((dp = readdir(fd)) != NULL)
		{
			
							std::vector<float> tmplt(512);
							
							std::string filename_fd = static_cast<std::string>(argv[1]) + "/" + static_cast<std::string>(dp->d_name);	
		
							if(filename_fd.find(".png")==std::string::npos)
								continue;
							cv::Mat img = cv::imread(filename_fd);
							std::cout << filename_fd << std::endl;
							
							cv::Mat original = img;
							int hght = original.rows;
							int wdth = original.cols;
							gpuprocess::Blob P_imgBuffer(hght, wdth, 3, sizeof(uint8_t));
							gpuprocess::Blob P_imgBuffer_transposed(hght, wdth, 3, sizeof(uint8_t));
							int resize_operation = 0;
							int padding_top = 0;
							int padding_side = 0;
							if (img.empty())
							{								
								/*for (auto t = 0; t < tmplt.size(); t++)
								{
									tmplt[t] = rand() / (float)RAND_MAX;
								}
								norm_tmpl(tmplt);

								for (auto t = 0; t < tmplt.size(); t++)
								{
									tmplt[t] *= 0.01f;
								}
								person_tmplts.emplace_back(hash_fn(id), tmplt);
								std::cout << "Image is corrupted!" << std::endl;*/
								continue;
							}
							else
								++total;
							cv::Mat img_resized = cv::Mat(cv::Size(640, 640), CV_32FC3);
							if (img.rows != 640 || img.cols != 640)
							{
								if (img.rows == img.cols && img.cols > 640)
								{
									cv::resize(img, img_resized, cv::Size(640, 640), 0, 0, 4);
									resize_operation = 1;
								}
								else if (img.rows <= 640 && img.cols >= 640)
								{
									cv::resize(img, img_resized, cv::Size(640, (int)((float)img.rows / (float)img.cols * 640)), 0, 0, 4);
									img = img_resized;
									int top = (int)(640 - img.rows) / 2;
									int bottom = top;
									if (bottom + top + img.rows < 640)
										bottom = top + 1;
									int left = 0;
									int right = left;
									cv::Scalar value(255, 255, 255);
									copyMakeBorder(img, img_resized, top, bottom, left, right, cv::BORDER_CONSTANT, value);
									resize_operation = 2;
									padding_top = top;
								}
								else if (img.rows >= 640 && img.cols <= 640)
								{
									cv::resize(img, img_resized, cv::Size((int)((float)img.cols / (float)img.rows * 640), 640), 0, 0, 4);
									img = img_resized;
									int left = (int)(640 - img.cols) / 2;
									int right = left;
									if (right + left + img.cols < 640)
										right = left + 1;
									int top = 0;
									int bottom = top;
									cv::Scalar value(255, 255, 255);
									copyMakeBorder(img, img_resized, top, bottom, left, right, cv::BORDER_CONSTANT, value);
									resize_operation = 3;
									padding_side = right;
								}
								else if (img.rows <= 640 && img.cols <= 640)
								{
									int left = (int)(640 - img.cols) / 2;
									int right = left;
									if (right + left + img.cols < 640)
										right = left + 1;
									int top = (int)(640 - img.rows) / 2;
									int bottom = top;
									if (bottom + top + img.rows < 640)
										bottom = top + 1;
									cv::Scalar value(255, 255, 255);
									copyMakeBorder(img, img_resized, top, bottom, left, right, cv::BORDER_CONSTANT, value);
									resize_operation = 4;
									padding_side = right;
									padding_top = top;
								}
								else if (img.rows >= 640 && img.cols >= 640 && img.cols != img.rows)
								{
									if (img.rows > img.cols)
									{
										cv::resize(img, img_resized, cv::Size((int)((float)img.cols / (float)img.rows * 640), 640), 0, 0, 4);
										img = img_resized;
										int left = (int)(640 - img.cols) / 2;
										int right = left;
										if (right + left + img.cols < 640)
											right = left + 1;
										int top = 0;
										int bottom = top;
										cv::Scalar value(255, 255, 255);
										copyMakeBorder(img, img_resized, top, bottom, left, right, cv::BORDER_CONSTANT, value);
										resize_operation = 5;
										padding_side = right;
									}
									if (img.rows < img.cols)
									{
										cv::resize(img, img_resized, cv::Size(640, (int)((float)img.rows / (float)img.cols * 640)), 0, 0, 4);
										img = img_resized;
										int top = (int)(640 - img.rows) / 2;
										int bottom = top;
										if (bottom + top + img.rows < 640)
											bottom = top + 1;
										int left = 0;
										int right = left;
										cv::Scalar value(255, 255, 255);
										copyMakeBorder(img, img_resized, top, bottom, left, right, cv::BORDER_CONSTANT, value);
										resize_operation = 6;
										padding_top = top;
									}
								}
								int height = img_resized.rows;
								int width = img_resized.cols;
							}
							else
							{
								img_resized = img;
								resize_operation = 1;
							}
							img = img_resized;

							//auto count = std::chrono::high_resolution_clock::now();
							//for (int i = 0; i < 1000; ++i)
							//{
							auto count = std::chrono::high_resolution_clock::now();
							gpuprocess::cpu_to_gpu(img.data, D_imgBuffer);

							gpuprocess::channel_swap(D_imgBuffer);
							gpuprocess::u8_to_f32(D_imgBuffer, D_imgBuffer_32f);
							float mean[]{ 104.0f, 117.0f, 123.0f };
							gpuprocess::subtract(D_imgBuffer_32f, mean);
							gpuprocess::to_planar(D_imgBuffer_32f, D_imgBuffer_transposed);

							inferenceDetector->inference(D_inputBuffer);
							//inferenceDetector->saveEngine("./engineDet");

							gpuprocess::gpu_to_cpu(D_detectionPoints, D_detPoints);
							gpuprocess::gpu_to_cpu(D_detectionRects, D_detRects);
							auto diff = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - count).count();
							all += diff;

							float* dett = nullptr;
							for (int i = 0; i < 1; ++i)
							{
								dett = D_detRects + i * 7;
								if (dett[2] < 0.2)
									break;
								//std::cout << "Label :" << dett[1] << "," << " confidence: " << dett[2]
								//	<< " xmin: " << dett[3] * 640
								//	<< " ymin: " << dett[4] * 640
								//	/*<< " xmax: " << (dett[5]+ dett[3]) * 640
								//	<< " ymax: " << (dett[6]+ dett[4])* 640*/
								//	<< " xmax: " << dett[5] * 640
								//	<< " ymax: " << dett[6] * 640
								//	<< std::endl;
								//cv::rectangle(img, cv::Point(dett[3] * 640, dett[4] * 640), cv::Point((dett[3] + dett[5]) * 640, (dett[4]+dett[6]) * 640),
								////~~~~~~~~~~~640_SIZE~~~~~~~~~~
								//cv::rectangle(img, cv::Point(dett[3] * 640, dett[4] * 640), cv::Point(dett[5] * 640, dett[6] * 640),
								//	cv::Scalar(0, 255, 0), 1);
								//cv::putText(img, std::to_string(dett[2]).substr(0, 6), cv::Point(dett[3] * 640, dett[4] * 640),
								//	CV_FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0, 255, 0), 0.5);
								////~~~~~~~~~~~~~~~~~~~~~~~
								//~~~~ORIG_SIZE~~~~~~~~
								int orig_xmin = 0;
								int orig_ymin = 0;
								int orig_xmax = 0;
								int orig_ymax = 0;
								//std::cout << "resize_operation: " << resize_operation << std::endl;
								switch (resize_operation)
								{
								case(1):
									orig_xmin = dett[3] * wdth;
									orig_ymin = dett[4] * hght;
									orig_xmax = dett[5] * wdth;
									orig_ymax = dett[6] * hght;
									break;
								case(2):
									orig_xmin = dett[3] * wdth;
									orig_ymin = (dett[4] / ((640 - 2 * (float)padding_top) / 640) - (float)padding_top / (640 - 2 * (float)padding_top))*hght;
									orig_xmax = dett[5] * wdth;
									orig_ymax = (dett[6] / ((640 - 2 * (float)padding_top) / 640) - (float)padding_top / (640 - 2 * (float)padding_top))*hght;
									break;
								case(3):
									orig_xmin = (dett[3] / ((640 - 2 * (float)padding_side) / 640) - (float)padding_side / (640 - 2 * (float)padding_side))*wdth;
									orig_ymin = dett[4] * hght;
									orig_xmax = (dett[5] / ((640 - 2 * (float)padding_side) / 640) - (float)padding_side / (640 - 2 * (float)padding_side))*wdth;
									orig_ymax = dett[6] * hght;
									break;
								case(4):									
									orig_xmin = (dett[3] * 640 / wdth - (float)padding_side / wdth)*wdth;
									orig_ymin = (dett[4] * 640 / hght - (float)padding_top / hght)*hght;
									orig_xmax = (dett[5] * 640 / wdth - (float)padding_side / wdth)*wdth;
									orig_ymax = (dett[6] * 640 / hght - (float)padding_top / hght)*hght;									
									break;
								case(5):
									orig_xmin = (dett[3] / ((640 - 2 * (float)padding_side) / 640) - (float)padding_side / (640 - 2 * (float)padding_side))*wdth;
									orig_ymin = dett[4] * hght;
									orig_xmax = (dett[5] / ((640 - 2 * (float)padding_side) / 640) - (float)padding_side / (640 - 2 * (float)padding_side))*wdth;
									orig_ymax = dett[6] * hght;
									break;
								case(6):
									orig_xmin = dett[3] * wdth;
									orig_ymin = (dett[4] / ((640 - 2 * (float)padding_top) / 640) - (float)padding_top / (640 - 2 * (float)padding_top))*hght;
									orig_xmax = dett[5] * wdth;
									orig_ymax = (dett[6] / ((640 - 2 * (float)padding_top) / 640) - (float)padding_top / (640 - 2 * (float)padding_top))*hght;
									break;
								default:
									std::cout << "Unknown transforantion! " << std::endl;
									break;
								}
								cv::rectangle(original, cv::Point(orig_xmin, orig_ymin), cv::Point(orig_xmax, orig_ymax),
									cv::Scalar(0, 255, 0), 1);
								// cv::putText(original, std::to_string(dett[2]).substr(0, 6), cv::Point(orig_xmin, orig_ymin),
								// 	CV_FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0, 255, 0), 0.5);
								//~~~~~~~~~~~~~~~~~~~~~
							}
							//Output to test ppoints detector
							int orig_x1 = 0;
							int orig_y1 = 0;
							int orig_x2 = 0;
							int orig_y2 = 0;
							int orig_x3 = 0;
							int orig_y3 = 0;
							int orig_x4 = 0;
							int orig_y4 = 0;
							int orig_x5 = 0;
							int orig_y5 = 0;
							float* dett_pnt = nullptr;
							for (int i = 0; i < 1; ++i)
							{
								dett_pnt = D_detPoints + i * 13;
								if (dett_pnt[2] < 0.2)
									break;
								/*std::cout << "label :" << dett_pnt[1] << "," << " confidence: " << dett_pnt[2]
									<< " xmin: " << dett_pnt[3] * 640
									<< " ymin: " << dett_pnt[4] * 640
									<< " xmax: " << dett_pnt[5] * 640
									<< " ymax: " << dett_pnt[6] * 640
									<< " xmin: " << dett_pnt[7] * 640
									<< " ymin: " << dett_pnt[8] * 640
									<< " xmax: " << dett_pnt[9] * 640
									<< " ymax: " << dett_pnt[10] * 640
									<< " xmin: " << dett_pnt[11] * 640
									<< " ymin: " << dett_pnt[12] * 640
									<< std::endl;*/
								////~~~~~~~640~~~~~~~~~~~
								//cv::rectangle(img, cv::Point(dett_pnt[3] * 640 - 1, dett_pnt[4] * 640 - 1), cv::Point(dett_pnt[3] * 640 + 1, dett_pnt[4] * 640 + 1), cv::Scalar(0, 255, 0), 1);
								//cv::rectangle(img, cv::Point(dett_pnt[5] * 640 - 1, dett_pnt[6] * 640 - 1), cv::Point(dett_pnt[5] * 640 + 1, dett_pnt[6] * 640 + 1), cv::Scalar(255, 255, 0), 1);
								//cv::rectangle(img, cv::Point(dett_pnt[7] * 640 - 1, dett_pnt[8] * 640 - 1), cv::Point(dett_pnt[7] * 640 + 1, dett_pnt[8] * 640 + 1), cv::Scalar(0, 255, 255), 1);
								//cv::rectangle(img, cv::Point(dett_pnt[9] * 640 - 1, dett_pnt[10] * 640 - 1), cv::Point(dett_pnt[9] * 640 + 1, dett_pnt[10] * 640 + 1), cv::Scalar(255, 0, 255), 1);
								//cv::rectangle(img, cv::Point(dett_pnt[11] * 640 - 1, dett_pnt[12] * 640 - 1), cv::Point(dett_pnt[11] * 640 + 1, dett_pnt[12] * 640 + 1), cv::Scalar(0, 0, 255), 1);
								////~~~~~~~~~~~~~~~~~~~~~
								//~~~~~~~~~ORIG_SIZE~~~~~~~~~~~~~
								switch (resize_operation)
								{
								case(1):
									orig_x1 = dett_pnt[3] * wdth;
									orig_y1 = dett_pnt[4] * hght;
									orig_x2 = dett_pnt[5] * wdth;
									orig_y2 = dett_pnt[6] * hght;
									orig_x3 = dett_pnt[7] * wdth;
									orig_y3 = dett_pnt[8] * hght;
									orig_x4 = dett_pnt[9] * wdth;
									orig_y4 = dett_pnt[10] * hght;
									orig_x5 = dett_pnt[11] * wdth;
									orig_y5 = dett_pnt[12] * hght;
									break;
								case(2):
									orig_x1 = dett_pnt[3] * wdth;
									orig_y1 = (dett_pnt[4] / ((640 - 2 * (float)padding_top) / 640) - (float)padding_top / (640 - 2 * (float)padding_top))*hght;
									orig_x2 = dett_pnt[5] * wdth;
									orig_y2 = (dett_pnt[6] / ((640 - 2 * (float)padding_top) / 640) - (float)padding_top / (640 - 2 * (float)padding_top))*hght;
									orig_x3 = dett_pnt[7] * wdth;
									orig_y3 = (dett_pnt[8] / ((640 - 2 * (float)padding_top) / 640) - (float)padding_top / (640 - 2 * (float)padding_top))*hght;
									orig_x4 = dett_pnt[9] * wdth;
									orig_y4 = (dett_pnt[10] / ((640 - 2 * (float)padding_top) / 640) - (float)padding_top / (640 - 2 * (float)padding_top))*hght;
									orig_x5 = dett_pnt[11] * wdth;
									orig_y5 = (dett_pnt[12] / ((640 - 2 * (float)padding_top) / 640) - (float)padding_top / (640 - 2 * (float)padding_top))*hght;
									break;
								case(3):
									orig_x1 = (dett_pnt[3] / ((640 - 2 * (float)padding_side) / 640) - (float)padding_side / (640 - 2 * (float)padding_side))*wdth;
									orig_y1 = dett_pnt[4] * hght;
									orig_x2 = (dett_pnt[5] / ((640 - 2 * (float)padding_side) / 640) - (float)padding_side / (640 - 2 * (float)padding_side))*wdth;
									orig_y2 = dett_pnt[6] * hght;
									orig_x3 = (dett_pnt[7] / ((640 - 2 * (float)padding_side) / 640) - (float)padding_side / (640 - 2 * (float)padding_side))*wdth;
									orig_y3 = dett_pnt[8] * hght;
									orig_x4 = (dett_pnt[9] / ((640 - 2 * (float)padding_side) / 640) - (float)padding_side / (640 - 2 * (float)padding_side))*wdth;
									orig_y4 = dett_pnt[10] * hght;
									orig_x5 = (dett_pnt[11] / ((640 - 2 * (float)padding_side) / 640) - (float)padding_side / (640 - 2 * (float)padding_side))*wdth;
									orig_y5 = dett_pnt[12] * hght;
									break;
								case(4):
									orig_x1 = (dett_pnt[3] * 640 / wdth - (float)padding_side / wdth)*wdth;
									orig_y1 = (dett_pnt[4] * 640 / hght - (float)padding_top / hght)*hght;
									orig_x2 = (dett_pnt[5] * 640 / wdth - (float)padding_side / wdth)*wdth;
									orig_y2 = (dett_pnt[6] * 640 / hght - (float)padding_top / hght)*hght;
									orig_x3 = (dett_pnt[7] * 640 / wdth - (float)padding_side / wdth)*wdth;
									orig_y3 = (dett_pnt[8] * 640 / hght - (float)padding_top / hght)*hght;
									orig_x4 = (dett_pnt[9] * 640 / wdth - (float)padding_side / wdth)*wdth;
									orig_y4 = (dett_pnt[10] * 640 / hght - (float)padding_top / hght)*hght;
									orig_x5 = (dett_pnt[11] * 640 / wdth - (float)padding_side / wdth)*wdth;
									orig_y5 = (dett_pnt[12] * 640 / hght - (float)padding_top / hght)*hght;
									break;
								case(5):
									orig_x1 = (dett_pnt[3] / ((640 - 2 * (float)padding_side) / 640) - (float)padding_side / (640 - 2 * (float)padding_side))*wdth;
									orig_y1 = dett_pnt[4] * hght;
									orig_x2 = (dett_pnt[5] / ((640 - 2 * (float)padding_side) / 640) - (float)padding_side / (640 - 2 * (float)padding_side))*wdth;
									orig_y2 = dett_pnt[6] * hght;
									orig_x3 = (dett_pnt[7] / ((640 - 2 * (float)padding_side) / 640) - (float)padding_side / (640 - 2 * (float)padding_side))*wdth;
									orig_y3 = dett_pnt[8] * hght;
									orig_x4 = (dett_pnt[9] / ((640 - 2 * (float)padding_side) / 640) - (float)padding_side / (640 - 2 * (float)padding_side))*wdth;
									orig_y4 = dett_pnt[10] * hght;
									orig_x5 = (dett_pnt[11] / ((640 - 2 * (float)padding_side) / 640) - (float)padding_side / (640 - 2 * (float)padding_side))*wdth;
									orig_y5 = dett_pnt[12] * hght;
									break;
								case(6):
									orig_x1 = dett_pnt[3] * wdth;
									orig_y1 = (dett_pnt[4] / ((640 - 2 * (float)padding_top) / 640) - (float)padding_top / (640 - 2 * (float)padding_top))*hght;
									orig_x2 = dett_pnt[5] * wdth;
									orig_y2 = (dett_pnt[6] / ((640 - 2 * (float)padding_top) / 640) - (float)padding_top / (640 - 2 * (float)padding_top))*hght;
									orig_x3 = dett_pnt[7] * wdth;
									orig_y3 = (dett_pnt[8] / ((640 - 2 * (float)padding_top) / 640) - (float)padding_top / (640 - 2 * (float)padding_top))*hght;
									orig_x4 = dett_pnt[9] * wdth;
									orig_y4 = (dett_pnt[10] / ((640 - 2 * (float)padding_top) / 640) - (float)padding_top / (640 - 2 * (float)padding_top))*hght;
									orig_x5 = dett_pnt[11] * wdth;
									orig_y5 = (dett_pnt[12] / ((640 - 2 * (float)padding_top) / 640) - (float)padding_top / (640 - 2 * (float)padding_top))*hght;
									break;
								default:
									std::cout << "Unknown transforantion! " << std::endl;
									break;
								}								
								cv::rectangle(original, cv::Point(orig_x1 - 1, orig_y1 - 1), cv::Point(orig_x1 + 1, orig_y1 + 1), cv::Scalar(0, 255, 0), 1);
								cv::rectangle(original, cv::Point(orig_x2 - 1, orig_y2 - 1), cv::Point(orig_x2 + 1, orig_y2 + 1), cv::Scalar(255, 255, 0), 1);
								cv::rectangle(original, cv::Point(orig_x3 - 1, orig_y3 - 1), cv::Point(orig_x3 + 1, orig_y3 + 1), cv::Scalar(0, 255, 255), 1);
								cv::rectangle(original, cv::Point(orig_x4 - 1, orig_y4 - 1), cv::Point(orig_x4 + 1, orig_y4 + 1), cv::Scalar(255, 0, 255), 1);
								cv::rectangle(original, cv::Point(orig_x5 - 1, orig_y5 - 1), cv::Point(orig_x5 + 1, orig_y5 + 1), cv::Scalar(0, 0, 255), 1);
								//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
								/*cv::rectangle(img, cv::Point(dett_pnt[3] * 640 - 1, dett_pnt[4] * 640 - 1), cv::Point(dett_pnt[3] * 640 + 1, dett_pnt[4] * 640 + 1), cv::Scalar(0, 255, 0), 1);
								cv::rectangle(img, cv::Point(dett_pnt[5] * 640 - 1, dett_pnt[6] * 640 - 1), cv::Point(dett_pnt[5] * 640 + 1, dett_pnt[6] * 640 + 1), cv::Scalar(255, 255, 0), 1);
								cv::rectangle(img, cv::Point(dett_pnt[7] * 640 - 1, dett_pnt[8] * 640 - 1), cv::Point(dett_pnt[7] * 640 + 1, dett_pnt[8] * 640 + 1), cv::Scalar(0, 255, 255), 1);
								cv::rectangle(img, cv::Point(dett_pnt[9] * 640 - 1, dett_pnt[10] * 640 - 1), cv::Point(dett_pnt[9] * 640 + 1, dett_pnt[10] * 640 + 1), cv::Scalar(255, 0, 255), 1);
								cv::rectangle(img, cv::Point(dett_pnt[11] * 640 - 1, dett_pnt[12] * 640 - 1), cv::Point(dett_pnt[11] * 640 + 1, dett_pnt[12] * 640 + 1), cv::Scalar(0, 0, 255), 1);*/
								//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
							}

							//auto path = std::experimental::filesystem::path("D:/test/Melanchon/new_det") / image_path.path().filename();
							//std::cout << path << std::endl;
							std::string res = "../res/" + static_cast<std::string>(dp->d_name);
							cv::imwrite(res, original);
							/*std::string p;
							if (cnt == 1)
								p = "D:/test/nouveau/face_found";
							else if (cnt > 1)
								p = "D:/test/nouveau/more_than_one_face_found";
							else if (cnt == 0)
								p = "D:/test/nouveau/face_not_found";
							if (dett[2] > 0.4 && dett[2] < 0.6 && cnt == 1)
							{
								p = "D:/test/nouveau/face_suspicious_confidence";
								auto path = std::experimental::filesystem::path(p) / entry.path().filename();
								cv::imwrite(path.generic_string(), img);
							}
							auto path = std::experimental::filesystem::path(p) / entry.path().filename();
							cv::imwrite(path.generic_string(), img);*/
							//cv::imshow("asfdasd", img);
							//cv::waitKey(0);

							//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
							// std::vector<Face> faces;
							// float* detector_pnts = nullptr;
							// for (int i = 0; i < 1; ++i)
							// {
							// 	detector_pnts = D_detPoints + i * 13;
							// 	if (detector_pnts[2] < 0.2)
							// 	{
							// 		std::cout << "can't detect face " << filename_fd << std::endl;
							// 		// for (auto t = 0; t < tmplt.size(); t++)
							// 		// {
							// 		// 	tmplt[t] = rand() / (float)RAND_MAX;
							// 		// }
							// 		// norm_tmpl(tmplt);

							// 		// for (auto t = 0; t < tmplt.size(); t++)
							// 		// {
							// 		// 	tmplt[t] *= 0.01f;
							// 		// }
							// 		// person_tmplts.emplace_back(hash_fn(id), tmplt);
							// 		break;
							// 	}
							// 	//####################
							// 	else
							// 	{												
							// 		/*faces.emplace_back(Face({ {dett_pnt[3], dett_pnt[4]}, {dett_pnt[5], dett_pnt[6]},
							// 			{dett_pnt[7], dett_pnt[8]}, {{dett_pnt[9], dett_pnt[10]}, {dett_pnt[11], dett_pnt[12]}} }));*/

							// 		gpuprocess::cpu_to_gpu(original.data, P_imgBuffer);
							// 		gpuprocess::channel_swap(P_imgBuffer);
							// 		gpuprocess::to_planar(P_imgBuffer, P_imgBuffer_transposed);

							// 		Face face = {{orig_x1, orig_y1},
							// 					{orig_x2, orig_y2},
							// 					{orig_x3, orig_y3},
							// 					{{orig_x4, orig_y4},
							// 					{orig_x5, orig_y5}} };

							// 		std::vector<void*> tempBufs({ P_imgBuffer_transposed.getDataPtr(), P_imgBuffer_resized.getDataPtr()});

							// 		//BaseFaceNormalizer norma;
							// 		//norma.transformImage(face, tempBufs, original.rows, original.cols);
							// 		//////~~~~~~~~~~~~~~
							// 		//norma.normalizeCPU(face, original.data, P_img112, original.cols, original.rows);
							// 		gpuprocess::cpu_to_gpu(P_img112, P_imgBuffer_resized);
							// 		//////~~~~~~~~~~~~~~
							// 		gpuprocess::u8_to_f32(P_imgBuffer_resized, P_imgBuffer_resized_32);

							// 		//inferenceDistinguish->inference(P_inputBuffer);

							// 		gpuprocess::gpu_to_cpu(P_detectionVector, P_detVector);

							// 		for (int i = 0; i < tmplt.size(); ++i)
							// 			tmplt[i] = *(P_detVector + i);
							// 		/*for (int i = 0; i < 20; ++i)
							// 			std::cout << *(P_detVector + i) << " ";*/
							// 		std::cout << std::endl;
									
							// 		// norm_tmpl(tmplt);
							// 		// person_tmplts.emplace_back(hash_fn(id), tmplt);
							// 		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
							// 	}
							// }							
							getchar();
		}

		//}
		std::cout << "Mean time per image: " << all / total << std::endl;
	}
	catch (std::exception &er)
	{
		std::cout << er.what() << std::endl;
	}
	closedir(fd);
	delete[] D_detPoints;
	delete[] D_detRects;
	delete[] P_detVector;
	delete[] P_img112;
	// std::cout << "detect count " << detect_count << std::endl;
	// int bin_count = 10000;
	
	// std::vector<size_t> diff(bin_count, 0);
	// std::vector<size_t> same(bin_count, 0);
	// size_t diff_count(0);
	// size_t same_count(0);
	
	// for (auto i = 0; i < person_tmplts.size(); ++i)
	// {
	// 	for (auto j = 0; j < person_tmplts.size(); ++j)
	// 	{
	// 		if (i == j) continue;
	
	// 		float score;

	// 		{
	// 			score = match_templates(person_tmplts[j].second, person_tmplts[i].second);
	// 		}
		
	// 		auto index = (int)(score * (bin_count - 1));
	// 		if (index > (bin_count - 1)) 
	// 			index = bin_count - 1;
	// 		if (index < 0) 
	// 			index = 0;
	
	// 		if (person_tmplts[i].first == person_tmplts[j].first)
	// 		{
	// 			same[index]++;
	// 			same_count++;
	// 		}
	// 		else
	// 		{
	// 			diff[index]++;
	// 			diff_count++;
	// 		}
	// 	}
	// }
	
	// std::vector<float> frr_hist(bin_count, .0f);
	// std::vector<float> far_hist(bin_count, .0f);
	
	// float v = 0;
	// for (auto i = 0; i < same.size(); ++i)
	// {
	// 	frr_hist[i] = v;
	// 	v = v + same[i] / (float)same_count;
	// }
	
	// v = 0;
	// for (auto i = 0; i < diff.size(); ++i)
	// {
	// 	v += diff[(bin_count - 1) - i] / (float)diff_count;
	// 	far_hist[(bin_count - 1) - i] = v;
	// }
	
	// for (auto i = 0; i < far_hist.size(); ++i)
	// {
	// 	if (far_hist[i] < 0.0001)
	// 	{
	// 		std::cout << frr_hist[i] << std::endl;
	// 		break;
	// 	}
	// }



	system("pause");
	return 0;
}

