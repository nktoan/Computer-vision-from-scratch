/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 20.05.2020 15:35:45
**/

#ifndef UTILS_H
#define UTILS_H
#define OPENCV_TRAITS_ENABLE_DEPRECATED

#include<iostream>
#include<cmath>
#include<vector>
#include<set>
#include<fstream>
#include"Libraries/Headers/opencv2/features2d/features2d.hpp"
#include"Libraries/Headers/opencv2/flann/flann.hpp"
#include"Libraries/Headers/opencv2/ml/ml.hpp"
#include"Libraries/Headers/opencv2/imgproc/imgproc.hpp"
#include"Libraries/Headers/opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define EPS 1e-6

Mat convertToGrayScale(const Mat &source);

void printMatrixInfo(const Mat &source);
float getValueOfMatrix(const Mat &source, int y, int x);
void setValueOfMatrix(Mat &source, int y, int x, float value);
float getMaxValueOfMatrix(const Mat &source);

Mat createSobelX();
Mat createSobelY();
Mat createGaussianKernel(int gaussianSize = 5, float signma = 1.0, bool divide = true, bool size_with_signma = false);
Mat createLoG_Kernel(int gaussianSize = 5, float signma = 1.0, bool normalized = false, bool size_with_signma = false);

Mat multiplyElementWise(const Mat& mat1, const Mat& mat2);
Mat mimusElementWise(const Mat& mat1, const Mat& mat2);

#endif