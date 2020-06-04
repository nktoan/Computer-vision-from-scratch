/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 22.05.2020 15:50:40
**/

#ifndef SIFT_H
#define SIFT_H

#include"Utils.h"

class myKeyPoint {
public:
	float y_image, x_image;
	float octave_index, layer_index;
	float hist_parabol_extrema;
	Mat feature;
public:
	myKeyPoint(int y, int x, int oct, int layer) :y_image(y), x_image(x), octave_index(oct), layer_index(layer) {}
	myKeyPoint(float y, float x, float oct, float layer, float hist_para_extre) :y_image(y), x_image(x), octave_index(oct), layer_index(layer), hist_parabol_extrema(hist_para_extre){}
};

class SiftDetector {
private:
	float getSignma(int oct, int layer, float signma, int num_octaves, int num_layers);

	vector<vector<Mat>> generate_gaussian_pyramid(const Mat &source, int n_octave, int n_scale_signma, float signma);
	vector<Mat> generate_gaussian_octave(const Mat &source, int n_octave, float &signma_change, float k);

	vector<vector<Mat>> generate_DOG_pyramid(const vector<vector<Mat>> &gaussian_pyramid);
	vector<Mat> generate_DOG_octave(const vector<Mat> &gaussian_octave);

	vector<myKeyPoint> extrema_detection_octave(int oct, const vector<vector<Mat>> &DoG_pyramid, const vector<vector<float>> &max_DoG_pyramid, int windowSize, float threshold_max = 0.3);
	vector<myKeyPoint> localise_keypoint_for_octave(int oct, const vector<vector<Mat>> &DoG_pyramid, const vector<vector<myKeyPoint>> &Extrema, float thresh_edge, float thresh_contrast, int window_size = 16);

	vector<myKeyPoint> orientation_assignment_for_octave(int oct, const vector<vector<Mat>> &DoG_pyramid, const vector<vector<myKeyPoint>> &keypoint, int num_bins);
	float fit_parabol(const vector<float> &hist, int mx_bin, int binwidth);

	void get_local_descriptors(int oct, const vector<vector<Mat>> &DoG_pyramid, vector<vector<myKeyPoint>> &orientation_keypoint, int window_size = 16, int num_subregion = 4, int num_bin = 8);
public:
	vector<vector<myKeyPoint>> siftDetector(const Mat &source, int num_octaves = 4, int num_scale_signma = 5, float signma = 1.6, float thresh_edge = 10, float thresh_contrast = 0.03, int windowSize = 16);
	
	void show_SIFT_key_points(const Mat &source, const vector<vector<myKeyPoint>> &sift_point, int octave = -1, int num_octaves = 4, bool wait_Key = true);
	void writingKeyPointToFile(const string &filename, const vector<vector<myKeyPoint>> &key_points);
	
	bool matchingTwoImages(const Mat &imgTrain, const Mat &imgTest, int octave = 1, float threshold_matching = 0.7, int vector_size = 128, bool is_show = true);
};


#endif