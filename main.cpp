/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 20.05.2020 15:40:30
**/

#include"Harris.h"
#include"Blob.h"
#include"Sift.h"

int main(int argc, char** argv)
{
	string image_name = "lena";
	string image_type = "png";

	Mat src = imread(image_name + '.' + image_type, IMREAD_COLOR);
	cout << "The input image information: ";
	printMatrixInfo(src);

	vector<vector<myKeyPoint>> key_points;

	SiftDetector siftDetector;
	key_points = siftDetector.siftDetector(src);
	string file_name = image_name + ".txt";

	siftDetector.writingKeyPointToFile(file_name, key_points);

	//siftDetector.siftDetector(src);

	//BlobDetector blobDetector;
	//HarrisDetector harrisDetector;

	//blobDetector.detectBlob(src);
	//harrisDetector.detectHarris(src);

	return 0;
}
