/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 20.05.2020 15:35:45
**/

#ifndef UTILS_H
#define UTILS_H

#include<iostream>
#include<cmath>
#include<vector>
#include<set>
#include"Libraries/Headers/opencv2/imgproc/imgproc.hpp"
#include"Libraries/Headers/opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

void printMatrixInfo(const Mat &source);
float getValueOfMatrix(const Mat &source, int y, int x);
void setValueOfMatrix(Mat &source, int y, int x, float value);

Mat createGaussianKernel(int GaussianSize = 5, float signma = 1.0);
Mat createSobelX();
Mat createSobelY();

Mat multiplyElementWise(const Mat& mat1, const Mat& mat2);

#endif