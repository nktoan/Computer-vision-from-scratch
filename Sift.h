/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 22.05.2020 15:50:40
**/

#ifndef SIFT_H
#define SIFT_H

#include"Utils.h"

class myKeyPoint {
public:
	int y_image, x_image;
	int octave_index, layer_index;
public:
	myKeyPoint(int y, int x, int oct, int layer) :y_image(y), x_image(x), octave_index(oct), layer_index(layer) {}
};

class SiftDetector {
private:
	vector<vector<Mat>> generate_gaussian_pyramid(const Mat &source, int n_octave, int n_scale_signma, float signma);
	vector<Mat> generate_gaussian_octave(const Mat &source, int n_octave, float &signma_change, float k);

	vector<vector<Mat>> generate_DOG_pyramid(const vector<vector<Mat>> &gaussian_pyramid);
	vector<Mat> generate_DOG_octave(const vector<Mat> &gaussian_octave);

	vector<myKeyPoint> extrema_detection_octave(int oct, const vector<vector<Mat>> &DoG_pyramid, const vector<vector<float>> &max_DoG_pyramid, int windowSize, float threshold_max = 0.3);
	vector<myKeyPoint> localise_keypoint_for_octave(int oct, const vector<vector<Mat>> &DoG_pyramid, const vector<vector<myKeyPoint>> &Extrema, float thresh_edge, float thresh_contrast);
public:
	void siftDetector(const Mat &source, int num_octaves = 4, int num_scale_signma = 5, float signma = 1.6, float thresh_edge = 10, float thresh_contrast = 0.03, int windowSize = 16);
};


#endif