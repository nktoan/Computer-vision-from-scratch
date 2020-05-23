/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 22.05.2020 15:51:40
**/

#include"Sift.h"

vector<vector<Mat>> SiftDetector::generate_gaussian_pyramid(const Mat &source, int n_octaves, int n_scale_signma, float signma) {
	vector<vector<Mat>> gaussian_pyramid;
	Mat src = source.clone();

	filter2D(src, src, CV_32FC1, createGaussianKernel(5, signma, true, true));

	float signma_change = signma, k = pow(2, 1.0 / (n_scale_signma - 1));

	for (int i = 0; i < n_octaves; ++i) {
		vector<Mat> octave = generate_gaussian_octave(src, n_scale_signma, signma_change, k);
		gaussian_pyramid.push_back(octave);

		assert(octave.size() - 1 >= 0);
		src = octave[octave.size() - 1];
		
		//signma_change = signma_change / (k * k);
		
		resize(src, src, cv::Size(), 0.5, 0.5, INTER_NEAREST);
	}
	//cout << "Size of pyramid: " << gaussian_pyramid.size() << endl;
	return gaussian_pyramid;
}

vector<Mat> SiftDetector::generate_gaussian_octave(const Mat &source, int n_scale_signma, float &signma_change, float k) {
	vector<Mat> octave;
	Mat src_base = source.clone();

	octave.push_back(src_base);

	for (int idx = 1; idx < n_scale_signma; ++idx) {
		signma_change = signma_change * k;
		Mat kernel = createGaussianKernel(5, signma_change, true, true);

		Mat next_level;
		filter2D(src_base, next_level, CV_32FC1, kernel);

		octave.push_back(next_level);
	}
	//cout << "Signma change at the end of octave : " << signma_change << endl;
	//cout << "Size of octave: " << octave.size() << endl;
	return octave;
}

vector<vector<Mat>> SiftDetector::generate_DOG_pyramid(const vector<vector<Mat>> &gaussian_pyramid) {
	vector<vector<Mat>> DoG_pyramid;
	for (vector<Mat> gaussian_octave : gaussian_pyramid)
		DoG_pyramid.push_back(generate_DOG_octave(gaussian_octave));
	return DoG_pyramid;
}

vector<Mat> SiftDetector::generate_DOG_octave(const vector<Mat> &gaussian_octave) {
	vector<Mat> DoG_octave;
	for (int i = 1; i < gaussian_octave.size(); ++i)
		DoG_octave.push_back(mimusElementWise(gaussian_octave[i], gaussian_octave[i - 1]));
	return DoG_octave;
}

vector<myKeyPoint> SiftDetector::extrema_detection_octave(int oct, const vector<vector<Mat>> &DoG_pyramid, const vector<vector<float>> &max_DoG_pyramid, int windowSize, float threshold_max) {
	vector<myKeyPoint> ktps_vec;
	for (int layer = 0; layer < DoG_pyramid[oct].size(); ++layer) {
		for (int y = windowSize / 2 + 1; y < DoG_pyramid[oct][layer].rows - windowSize / 2 - 1; ++y) {
			for (int x = windowSize / 2 + 1; x < DoG_pyramid[oct][layer].cols - windowSize / 2 - 1; ++x) {
				float val = getValueOfMatrix(DoG_pyramid[oct][layer], y, x);
				float val_squared = val * val;

				if (val_squared < threshold_max * max_DoG_pyramid[oct][layer]) continue;
				bool found_peak = true;

				for (int step_idx = -1; step_idx <= 1; ++step_idx) {
					if (found_peak == false) break;

					for (int step_x = -1; step_x <= 1; ++step_x) {
						if (found_peak == false) break;

						for (int step_y = -1; step_y <= 1; ++step_y) {
							if (found_peak == false) break;

							int cur_layer = layer + step_idx, cur_y = y + step_y, cur_x = x + step_x;
							if (cur_layer >= DoG_pyramid[oct].size() || cur_layer < 0) continue;

							float neightbor_squared = getValueOfMatrix(DoG_pyramid[oct][cur_layer], cur_y, cur_x) * getValueOfMatrix(DoG_pyramid[oct][cur_layer], cur_y, cur_x);

							if (val_squared < neightbor_squared)
								found_peak = false;
						}
					}
				}

				if (found_peak == true)
					ktps_vec.push_back(myKeyPoint(y, x, oct, layer));
			}
		}
	}
	return ktps_vec;
}

vector<myKeyPoint> SiftDetector::localise_keypoint_for_octave(int oct, const vector<vector<Mat>> &DoG_pyramid, const vector<vector<myKeyPoint>> &Extrema, float thresh_edge, float thresh_contrast){
	vector<myKeyPoint> key_points;
	int n_layer = DoG_pyramid[oct].size();

	for (myKeyPoint point : Extrema[oct]) {
		int y = point.y_image, x = point.x_image, s = point.layer_index;

		float dx, dy, ds, dxx, dxy, dxs, dyy, dys, dss;
		dx = (getValueOfMatrix(DoG_pyramid[oct][s], y, x + 1) - getValueOfMatrix(DoG_pyramid[oct][s], y, x - 1)) / 2.0;
		dy = (getValueOfMatrix(DoG_pyramid[oct][s], y + 1, x) - getValueOfMatrix(DoG_pyramid[oct][s], y - 1, x)) / 2.0;
		dxx = getValueOfMatrix(DoG_pyramid[oct][s], y, x + 1) - 2.0 * getValueOfMatrix(DoG_pyramid[oct][s], y, x) + getValueOfMatrix(DoG_pyramid[oct][s], y, x - 1);
		dyy = getValueOfMatrix(DoG_pyramid[oct][s], y + 1, x) - 2.0 * getValueOfMatrix(DoG_pyramid[oct][s], y, x) + getValueOfMatrix(DoG_pyramid[oct][s], y - 1, x);
		dxy = ((getValueOfMatrix(DoG_pyramid[oct][s], y + 1, x + 1) - getValueOfMatrix(DoG_pyramid[oct][s], y + 1, x - 1))
				- (getValueOfMatrix(DoG_pyramid[oct][s], y - 1, x + 1) - getValueOfMatrix(DoG_pyramid[oct][s], y - 1, x - 1))) / 4.0;
		if (s + 1 >= n_layer) {
			ds = -1.0 * getValueOfMatrix(DoG_pyramid[oct][s - 1], y, x) / 2.0;
			dxs = -1.0 * (getValueOfMatrix(DoG_pyramid[oct][s - 1], y, x + 1) - getValueOfMatrix(DoG_pyramid[oct][s - 1], y, x - 1)) / 4.0;
			dys = -1.0 * (getValueOfMatrix(DoG_pyramid[oct][s - 1], y + 1, x) - getValueOfMatrix(DoG_pyramid[oct][s - 1], y - 1, x)) / 4.0;
			dss =  - 2.0 * getValueOfMatrix(DoG_pyramid[oct][s], y, x) + getValueOfMatrix(DoG_pyramid[oct][s - 1], y, x);
		}
		else if (s - 1 < 0) {
			ds = getValueOfMatrix(DoG_pyramid[oct][s + 1], y, x) / 2.0;
			dxs = (getValueOfMatrix(DoG_pyramid[oct][s + 1], y, x + 1) - getValueOfMatrix(DoG_pyramid[oct][s + 1], y, x - 1)) / 4.0;
			dys = (getValueOfMatrix(DoG_pyramid[oct][s + 1], y + 1, x) - getValueOfMatrix(DoG_pyramid[oct][s + 1], y - 1, x)) / 4.0;
			dss = getValueOfMatrix(DoG_pyramid[oct][s + 1], y, x) - 2.0 * getValueOfMatrix(DoG_pyramid[oct][s], y, x);
		}
		else {
			ds = (getValueOfMatrix(DoG_pyramid[oct][(s + 1)], y, x) - getValueOfMatrix(DoG_pyramid[oct][s - 1], y, x)) / 2.0;
			dxs = ((getValueOfMatrix(DoG_pyramid[oct][s + 1], y, x + 1) - getValueOfMatrix(DoG_pyramid[oct][s + 1], y, x - 1))
				- (getValueOfMatrix(DoG_pyramid[oct][s - 1], y, x + 1) - getValueOfMatrix(DoG_pyramid[oct][s - 1], y, x - 1))) / 4.0;
			dys = ((getValueOfMatrix(DoG_pyramid[oct][s + 1], y + 1, x) - getValueOfMatrix(DoG_pyramid[oct][s + 1], y - 1, x))
				- (getValueOfMatrix(DoG_pyramid[oct][s - 1], y + 1, x) - getValueOfMatrix(DoG_pyramid[oct][s - 1], y - 1, x))) / 4.0;
			dss = getValueOfMatrix(DoG_pyramid[oct][s + 1], y, x) - 2.0 * getValueOfMatrix(DoG_pyramid[oct][s], y, x) + getValueOfMatrix(DoG_pyramid[oct][s - 1], y, x);
		}
		Mat HessianDoG = Mat::zeros(3, 3, CV_32FC1);
		HessianDoG.at<float>(0, 0) = dxx; HessianDoG.at<float>(0, 1) = dxy; HessianDoG.at<float>(0, 2) = dxs;
		HessianDoG.at<float>(1, 0) = dxy; HessianDoG.at<float>(1, 1) = dyy; HessianDoG.at<float>(1, 2) = dys;
		HessianDoG.at<float>(2, 0) = dxs; HessianDoG.at<float>(2, 1) = dys; HessianDoG.at<float>(2, 2) = dss;

		Mat J = Mat::zeros(3, 1, CV_32FC1);
		J.at<float>(0, 0) = dx; J.at<float>(1, 0) = dy; J.at<float>(2, 0) = ds;

		Mat offset_mat = -(HessianDoG.inv()) * (J);

		float contrast = getValueOfMatrix(DoG_pyramid[oct][s], y, x) + 0.5*(J.dot(offset_mat));

		if (abs(contrast) < thresh_contrast) continue;

		Mat H = Mat::zeros(2, 2, CV_32FC1);
		H.at<float>(0, 0) = dxx; H.at<float>(0, 1) = dxy;
		H.at<float>(1, 0) = dxy; H.at<float>(1, 1) = dyy;

		float trace_H = H.at<float>(0, 0) + H.at<float>(1, 1);
		float det_H = H.at<float>(0, 0) * H.at<float>(1, 1) - H.at<float>(0, 1) * H.at<float>(0, 1);

		float ratio = 1.0 * trace_H * trace_H / det_H;
		if (ratio > (1.0*(thresh_contrast + 1)*(thresh_contrast + 1) / thresh_contrast)) continue;

		key_points.push_back(point);
	}
	return key_points;
}

void SiftDetector::siftDetector(const Mat &source, int num_octaves, int num_scale_signma, float signma, float thresh_edge, float thresh_contrast, int windowSize){
	Mat image = source.clone();
	/* Step 1: Convert Image to GrayScale and Double the Image */
	Mat srcGray = convertToGrayScale(image);
	
	resize(srcGray, srcGray, cv::Size(), 2.0, 2.0, INTER_LINEAR);
	
	/* Step 2: Generate Gaussian Pyramid */
	vector<vector<Mat>> gaussian_pyramid = generate_gaussian_pyramid(srcGray, num_octaves, num_scale_signma, signma);

	/* Step 3: Generate DoG Pyramid and find Maximum of DoG Pyradmid for heuristic */
	vector<vector<Mat>> DoG_pyramid = generate_DOG_pyramid(gaussian_pyramid);

	vector<vector<float>> max_squared_DoG_pyramid(num_octaves, vector<float>(num_scale_signma - 1, -1));

	for (int i = 0; i < num_octaves; ++i)
		for (int j = 0; j < num_scale_signma - 1; ++j)
			for (int y = 0; y < DoG_pyramid[i][j].rows; ++y)
				for (int x = 0; x < DoG_pyramid[i][j].cols; ++x)
					max_squared_DoG_pyramid[i][j] = max(max_squared_DoG_pyramid[i][j], getValueOfMatrix(DoG_pyramid[i][j], y, x) * getValueOfMatrix(DoG_pyramid[i][j], y, x));

	/* Step 4: Extrema Detection */
	vector<vector<myKeyPoint>> Extrema(num_octaves);
	float threshold_max = 0.3;

	for (int oct = 0; oct < num_octaves; ++oct)
		for (myKeyPoint kpts : extrema_detection_octave(oct, DoG_pyramid, max_squared_DoG_pyramid, windowSize, threshold_max))
			Extrema[oct].push_back(kpts);

	/* Step 5: Keypoint Localization */
	vector<vector<myKeyPoint>> keyPoints(num_octaves);

	for (int oct = 0; oct < num_octaves; ++oct)
		for (myKeyPoint kpts : localise_keypoint_for_octave(oct, DoG_pyramid, Extrema, thresh_edge, thresh_contrast))
			keyPoints[oct].push_back(kpts);

	/* Step 6: Orientation Assignment */

	/* Step n: Draw the KeyPoints */
	Mat dst = source.clone();
	for (int oct = 0; oct < num_octaves; ++oct) {
		for (myKeyPoint points : keyPoints[oct]) {
			circle(dst, Point(points.x_image, points.y_image), sqrt(2), Scalar(0, 0, 255), 2, 8, 0);//(y,x) -> Point(x,y)
			//cout << points.x_image << " " << points.y_image << endl;
		}
		cout << "Choose : " << keyPoints[oct].size() << " points from octave " << oct << endl;
	}
	/* Step n+1: Show the KeyPoints image */
	namedWindow("Sift_Detector");
	imshow("Sift_Detector", dst);
	waitKey(0);

}