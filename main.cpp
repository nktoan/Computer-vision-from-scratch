/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 20.05.2020 15:40:30
**/

#include"ImageOperator.h"

int main(int argc, char** argv)
{
	Mat src = imread("sunflower_small.jpg", IMREAD_COLOR);
	detectDOG(src);
	

	return 0;
}
