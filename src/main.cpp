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
#include "gpuprocess.h"


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
		"../model.onnx", 1);
	delete instance;
	
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
		int id = 1;
		while ((dp = readdir(fd)) != NULL)
		{
										
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
								continue;
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

							auto count = std::chrono::high_resolution_clock::now();
							gpuprocess::cpu_to_gpu(img.data, D_imgBuffer);

							gpuprocess::channel_swap(D_imgBuffer);
							gpuprocess::u8_to_f32(D_imgBuffer, D_imgBuffer_32f);
							float mean[]{ 104.0f, 117.0f, 123.0f };
							gpuprocess::subtract(D_imgBuffer_32f, mean);
							gpuprocess::to_planar(D_imgBuffer_32f, D_imgBuffer_transposed);

							inferenceDetector->inference(D_inputBuffer);

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
								
								int orig_xmin = 0;
								int orig_ymin = 0;
								int orig_xmax = 0;
								int orig_ymax = 0;
								
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
							}
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
							}				
				
							getchar();
		}

		std::cout << "Mean time per image: " << all / total << std::endl;
	}
	catch (std::exception &er)
	{
		std::cout << er.what() << std::endl;
	}
	closedir(fd);
	delete[] D_detPoints;
	delete[] D_detRects;
	
	system("pause");
	return 0;
}

