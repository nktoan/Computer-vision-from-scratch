/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 20.05.2020 15:40:30
**/

#include"Harris.h"
#include"Blob.h"
#include"Sift.h"

void execute(int argc, const vector<string> &argv);
void help();

int main(int argc, char** argv)
{
	vector<string> argv_str(argc);
	for (int i = 0; i < argc; ++i)
		argv_str[i] = argv[i];

	execute(argc, argv_str);

	return 0;
}

void execute(int argc, const vector<string> &argv) {
	/*
	Ma lenh: 
	+ Harris: "[ten_chuong_trinh] [duong_dan_anh] harris [tham_so_k](vd: 0.05) [tham_so_alpha](vd: 0.01)"
	+ Blob: "[ten_chuong_trinh] [duong_dan_anh] blob [tham_so_threshold_max](vd: 0.3) [tham_so_signma](vd: 1.0) [tham_so_k](vd: 1.41421356)"
	+ Dog: "[ten_chuong_trinh] [duong_dan_anh] dog [tham_so_threshold_max](vd: 0.3) [tham_so_signma](vd: 1.0) [tham_so_k](vd: 1.41421356)"
	+ Sift: "[ten_chuong_trinh] [duong_dan_anh] sift [n_octave](vd: 4) [n_scales](vd: 5) [tham_so_signma](vd: 1.6)"
	+ Matching: "[ten_chuong_trinh] [duong_dan_anh_train] matching [duong_dan_anh_test] [tham_so_octave](vd: 1)"
	*/
	if (argc < 2)
		help();
	else {
		Mat srcImg = imread(argv[1], IMREAD_COLOR);
		if (srcImg.empty()) {
			cout << "Khong tim thay anh argv[1] " << endl;
			return;
		}
		cout << "The argv[1] image information: ";
		printMatrixInfo(srcImg); //cout << endl;


		if (argc == 2) {
			imshow("Source_Image", srcImg);

			help(); return;
		}
		if (argv[2] == "harris") {
			imshow("Source_Image", srcImg);

			HarrisDetector harrisDetector;
			float k, alpha;

			(argv.size() >= 4) ? (k = stof(argv[3])) : (k = 0.05);
			(argv.size() >= 5) ? (alpha = stof(argv[4])) : (alpha = 0.01);

			vector<CornerPoint> cornerPoints = harrisDetector.detectHarris(srcImg, k, alpha);
			harrisDetector.showCornerPoint(srcImg, cornerPoints);
		}
		else if (argv[2] == "blob" || argv[2] == "dog") {
			imshow("Source_Image", srcImg);

			BlobDetector blobDetector;
			float k, signma, threshold_max;
			set<tuple<int, int, float>> blobPoints;

			(argv.size() >= 4) ? (threshold_max = stof(argv[3])) : (threshold_max = 0.3);
			(argv.size() >= 5) ? (signma = stof(argv[4])) : (signma = 1.0);
			(argv.size() >= 6) ? (k = stof(argv[5])) : (k = sqrt(2));

			if (argv[2] == "blob") {
				blobPoints = blobDetector.detectBlob(srcImg, signma, k, threshold_max);
				blobDetector.showBlobPoint_BlobDetector(srcImg, blobPoints);
			}
			else {
				blobPoints = blobDetector.detectDOG(srcImg, signma, k, threshold_max);
				blobDetector.showBlobPoint_DOGDetector(srcImg, blobPoints);
			}
		}
		else if (argv[2] == "sift") {
			imshow("Source_Image", srcImg);

			SiftDetector siftDetector;
			int n_octave, n_scales;
			float signma;

			(argv.size() >= 4) ? (n_octave = stoi(argv[3])) : (n_octave = 4);
			(argv.size() >= 5) ? (n_scales = stoi(argv[4])) : (n_scales = 5);
			(argv.size() >= 6) ? (signma = stof(argv[5])) : (signma = 1.6);

			if (n_octave < 2) n_octave += (2 - n_octave);
			if (n_scales < 3) n_scales += (3 - n_scales);

			vector<vector<myKeyPoint>> key_points = siftDetector.siftDetector(srcImg, n_octave, n_scales, signma);
			siftDetector.show_SIFT_key_points(srcImg, key_points, 1, n_octave);
		}
		else if (argv[2] == "matching") {
			int octave;
			SiftDetector siftDetector;

			(argv.size() >= 5) ? (octave = stoi(argv[4])) : (octave = 1);

			Mat testImg = imread(argv[3], IMREAD_COLOR);
			if (testImg.empty()) {
				cout << "Khong tim thay anh argv[3] " << endl;
				return;
			}
			cout << "The argv[3] image information: ";
			printMatrixInfo(testImg); cout << endl;

			if (siftDetector.matchingTwoImages(srcImg, testImg, octave))
				cout << "Matching" << endl;
		}
		else
			help();
	}
}

void help() {
	cout << "Huong dan su dung:" << endl;
	cout << "Harris: " << "[ten_chuong_trinh] [duong_dan_anh] harris [tham_so_k](vd: 0.05) [tham_so_alpha](vd: 0.01)" << endl;
	cout << "Blob: " << "[ten_chuong_trinh] [duong_dan_anh] blob [tham_so_threshold_max](vd: 0.3) [tham_so_signma](vd: 1.0) [tham_so_k](vd: 1.41421356)" << endl;
	cout << "Dog: " << "[ten_chuong_trinh] [duong_dan_anh] dog [tham_so_threshold_max](vd: 0.3) [tham_so_signma](vd: 1.0) [tham_so_k](vd: 1.41421356)" << endl;
	cout << "Sift: " << "[ten_chuong_trinh] [duong_dan_anh] sift [n_octave](vd: 4) [n_scales](vd: 5) [tham_so_signma](vd: 1.6)" << endl;
	cout << "Matching: " << "[ten_chuong_trinh] [duong_dan_anh_train] matching [duong_dan_anh_test] [tham_so_octave](vd: 1)" << endl;
}