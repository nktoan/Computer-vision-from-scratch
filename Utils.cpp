/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 20.05.2020 15:40:45
**/

#include"Utils.h"

Mat convertToGrayScale(const Mat &source) {
	Mat dst;
	cvtColor(source, dst, COLOR_BGR2GRAY);
	return dst;
}

void printMatrixInfo(const Mat &source) {
	int typeMatrix = source.type();
	string printOut;

	uchar depth = typeMatrix & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (typeMatrix >> CV_CN_SHIFT);

	switch (depth) {
		case CV_8U:  printOut = "8U"; break;
		case CV_8S:  printOut = "8S"; break;
		case CV_16U: printOut = "16U"; break;
		case CV_16S: printOut = "16S"; break;
		case CV_32S: printOut = "32S"; break;
		case CV_32F: printOut = "32F"; break;
		case CV_64F: printOut = "64F"; break;
		default:     printOut = "User"; break;
	}

	printOut += "C";
	printOut += (chans + '0');

	cout << printOut << " " << source.rows << "x" << source.cols << endl;
}

float getValueOfMatrix(const Mat &source, int y, int x) {
	int typeMatrix = source.type();
	uchar depth = typeMatrix & CV_MAT_DEPTH_MASK;

	switch (depth) {
		//case CV_8U:  return (float)source.at<uchar>(y, x);
		case CV_32F: return source.at<float>(y, x); break;
		default:     return (float)source.at<uchar>(y, x); break;
	}
}

void setValueOfMatrix(Mat &source, int y, int x, float value) {
	int typeMatrix = source.type();
	uchar depth = typeMatrix & CV_MAT_DEPTH_MASK;

	switch (depth) {
		case CV_32F: source.at<float>(y, x) = value; break;
		default:     source.at<uchar>(y, x) = (uchar)value; break;
	}
}

float getMaxValueOfMatrix(const Mat &source){
	float mx = -1e9;
	for (int y = 0; y < source.rows; ++y) 
		for (int x = 0; x < source.cols; ++x) 
			mx = max(mx, getValueOfMatrix(source, y, x));
	return mx;
}

Mat createGaussianKernel(int gaussianSize, float signma, bool divide, bool size_with_signma){
	if (size_with_signma == false)
		assert(gaussianSize % 2 == 1);
	else
		gaussianSize = (int) 2 * ceil(3 * signma) + 1;

	Mat gaussianKernel = Mat::zeros(gaussianSize, gaussianSize, CV_32FC1);
	float sum = 0.0;
	float var = 2 * signma * signma;
	float r, pi = 2 * acos(0);

	for (int y = -(gaussianSize / 2); y <= gaussianSize / 2; ++y) {
		for (int x = -(gaussianSize / 2); x <= gaussianSize / 2; ++x) {
			r = sqrt(x*x + y*y);
			gaussianKernel.at<float>(y + gaussianSize / 2, x + gaussianSize / 2) = exp(-(r*r) / var) / (pi*var);
			sum += gaussianKernel.at<float>(y + gaussianSize / 2, x + gaussianSize / 2);
		}
	}
	if (divide == true) {
		for (int i = 0; i < gaussianSize; ++i)
			for (int j = 0; j < gaussianSize; ++j)
				gaussianKernel.at<float>(i, j) = gaussianKernel.at<float>(i, j) * 1.0 / sum;
	}

	return gaussianKernel;
}
Mat createSobelX() {
	return (Mat_<float>(3, 3) << -1, 0, 1,
		-2, 0, 2,
		-1, 0, 1);
}
Mat createSobelY() {
	return (Mat_<float>(3, 3) << -1, -2, -1,
		0, 0, 0,
		1, 2, 1);
}

Mat multiplyElementWise(const Mat& mat1, const Mat& mat2) {
	assert(mat1.rows == mat2.rows && mat1.cols == mat2.cols);

	int height = mat1.rows, width = mat2.cols;
	Mat res = mat1.clone();
	
	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x) {
			float multiply_res = getValueOfMatrix(mat1, y, x) * getValueOfMatrix(mat2, y, x);
			setValueOfMatrix(res, y, x, multiply_res);
		}

	return res;
}

Mat mimusElementWise(const Mat& mat1, const Mat& mat2) {
	assert(mat1.rows == mat2.rows && mat1.cols == mat2.cols);

	int height = mat1.rows, width = mat2.cols;
	Mat res = mat1.clone();

	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x) {
			float multiply_res = getValueOfMatrix(mat1, y, x) - getValueOfMatrix(mat2, y, x);
			setValueOfMatrix(res, y, x, multiply_res);
		}

	return res;
}

Mat createLoG_Kernel(int gaussianSize, float signma, bool normalized){
	assert(gaussianSize % 2 == 1);
	Mat LoG_kernel = Mat::zeros(gaussianSize, gaussianSize, CV_32FC1);
	float sum = 0.0;
	float var = 2 * signma * signma;
	float r, pi = 2 * acos(0);	
	
	for (int y = -(gaussianSize / 2); y <= gaussianSize / 2; ++y) {
		for (int x = -(gaussianSize / 2); x <= gaussianSize / 2; ++x) {
			r = sqrt(x*x + y*y);
			float val = (-4.0) * (exp(-(r * r) / var) * (1.0 - (r * r / var))) / (pi * var * var);
			if (normalized == true) val = val * signma * signma;
			LoG_kernel.at<float>(y + gaussianSize / 2, x + gaussianSize / 2) = val;
			sum += LoG_kernel.at<float>(y + gaussianSize / 2, x + gaussianSize / 2);
		}
	}

	return LoG_kernel;
}