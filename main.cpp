/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 20.05.2020 15:40:30
**/

#include"ImageOperator.h"

int main(int argc, char** argv)
{
	Mat src = imread("sunflower.jpeg", IMREAD_COLOR);
	detectBlob(src);
	
	/*
	Mat kernel = createLoG_Kernel(5, 1.0);
	for (int y = 0; y < kernel.rows; ++y) {
		for (int x = 0; x < kernel.cols; ++x)
			cout << kernel.at<float>(y, x) << " ";
		cout << endl;
	}
	*/
	return 0;
}
