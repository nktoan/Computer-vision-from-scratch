/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 20.05.2020 15:40:30
**/

#include"Harris.h"
#include"Blob.h"
#include"Sift.h"

int main(int argc, char** argv)
{
	Mat src = imread("sunflower_small.jpg", IMREAD_COLOR);
	printMatrixInfo(src);

	SiftDetector siftDetector;
	siftDetector.siftDetector(src);
	//BlobDetector blobDetector;
	//HarrisDetector harrisDetector;

	//blobDetector.detectBlob(src);
	//harrisDetector.detectHarris(src);

	return 0;
}
