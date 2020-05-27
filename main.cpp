/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 20.05.2020 15:40:30
**/

#include"Harris.h"
#include"Blob.h"
#include"Sift.h"

int main(int argc, char** argv)
{
	string image_name = "sunflower_small";
	string image_type = "jpg";

	Mat src = imread(image_name + '.' + image_type, IMREAD_COLOR);
	cout << "The input image information: ";
	printMatrixInfo(src);

	BlobDetector blobDetector;
	HarrisDetector harrisDetector;
	SiftDetector siftDetector;

	//bool wait_key = false;
	/* 1. Detect Corner using Harris Detector */
	harrisDetector.detectHarris(src);

	/* 2. Detect Blob using Blob, DoG Detector (blob slace space detection) */
	blobDetector.detectBlob(src);

	blobDetector.detectDOG(src);

	/* 3. Extrace SIFT features from Image */
	vector<vector<myKeyPoint>> key_points;	
	key_points = siftDetector.siftDetector(src);

	string file_name = image_name + ".txt";
	siftDetector.writingKeyPointToFile(file_name, key_points);

	return 0;
}
