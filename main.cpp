/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 20.05.2020 15:40:30
**/

#include"Harris.h"
#include"Blob.h"
#include"Sift.h"

int main(int argc, char** argv)
{
	string image_name_train = "01_1", image_name_test = "01";
	string image_type_train = "jpg", image_type_test = "jpg";

	Mat src_train = imread(image_name_train + '.' + image_type_train, IMREAD_COLOR);
	cout << "The input train_image information: ";
	printMatrixInfo(src_train);

	Mat src_test = imread(image_name_test + '.' + image_type_test, IMREAD_COLOR);
	cout << "The input test_image information: ";
	printMatrixInfo(src_test);

	BlobDetector blobDetector;
	HarrisDetector harrisDetector;
	SiftDetector siftDetector;

	/* 1. Detect Corner using Harris Detector */
	//harrisDetector.detectHarris(src);

	/* 2. Detect Blob using Blob, DoG Detector (blob slace space detection) */
	//blobDetector.detectBlob(src);

	//blobDetector.detectDOG(src);

	/* 3. Extrace SIFT features from Image */
	siftDetector.matchingTwoImages(src_train, src_test);

	return 0;
}
