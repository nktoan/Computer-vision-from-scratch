/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 22.05.2020 15:51:40
**/

#include"Sift.h"

vector<vector<Mat>> SiftDetector::generate_gaussian_pyramid(const Mat &source, int n_octaves, int n_scale_signma, float signma) {
	vector<vector<Mat>> gaussian_pyramid;
	Mat src = source.clone();

	float signma_change = signma, k = sqrt(2);

	for (int i = 0; i < n_octaves; ++i) {
		vector<Mat> octave = generate_gaussian_octave(src, n_scale_signma, signma_change, k);
		gaussian_pyramid.push_back(octave);

		src = source.clone();
		
		//signma_change = signma_change / (k * k);
		float scale_x, scale_y;
		scale_x = scale_y = pow(0.5, i + 1);
		resize(src, src, cv::Size(), scale_x, scale_y, INTER_NEAREST);

		signma_change = signma;
	}
	//cout << "Size of pyramid: " << gaussian_pyramid.size() << endl;
	return gaussian_pyramid;
}

vector<Mat> SiftDetector::generate_gaussian_octave(const Mat &source, int n_scale_signma, float &signma_change, float k) {
	vector<Mat> octave;
	Mat src_base = source.clone();

	//octave.push_back(src_base);

	for (int idx = 0; idx < n_scale_signma; ++idx) {
		Mat kernel = createGaussianKernel(5, signma_change, true, true);

		Mat next_level;
		filter2D(src_base, next_level, CV_32FC1, kernel);
		octave.push_back(next_level);

		signma_change = signma_change * k;
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
		int y = (int)point.y_image, x = (int)point.x_image, s = (int)point.layer_index;

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
		float det_H = H.at<float>(0, 0) * H.at<float>(1, 1) - H.at<float>(0, 1) * H.at<float>(1, 0);

		float ratio = 1.0 * trace_H * trace_H / det_H;
		if (ratio > (1.0*(thresh_edge + 1)*(thresh_edge + 1) / thresh_edge)) continue;

		point.x_image += offset_mat.at<float>(0, 0);
		point.y_image += offset_mat.at<float>(1, 0);
		point.layer_index += offset_mat.at<float>(2, 0);

		key_points.push_back(point);
	}
	return key_points;
}

float SiftDetector::getSignma(int oct, int layer, float signma, int num_octaves, int num_layers) {
	/*
	vector<vector<float>> signma_vect(oct, vector<float>(layer, 0));
	float signma_change = signma, k = pow(2, 1.0 / (num_layers - 1));
	signma_vect[0][0] = signma;

	for (int i = 0; i < num_octaves; ++i)
		for (int j = 0; j < num_layers; ++j) {
			signma_change = (i == 0 && j == 0) ? (signma_change) : signma_change * k;
			signma_vect[i][j] = signma_change;
		}

	return signma_vect[oct][layer];
	*/
	return pow(sqrt(2), layer) * signma;
}

vector<myKeyPoint> SiftDetector::orientation_assignment_for_octave(int oct, const vector<vector<Mat>> &DoG_pyramid, const vector<vector<myKeyPoint>> &keypoint, int num_bins) {
	vector<myKeyPoint> orientation_kpts;

	int n_layer = DoG_pyramid[oct].size(), binwidth = 360 / num_bins;
	for (myKeyPoint point : keypoint[oct]) {
	
		int y = (int) point.y_image, x = (int) point.x_image, s = (int)point.layer_index;
		(s < 0) ? (s = 0) : ((s >= n_layer) ? (s = n_layer - 1) : s);
		float signma = (s == 0) ? 1.0 : (s * 1.5);
		int w = int(2 * ceil(signma) + 1);
		Mat kernel = createGaussianKernel((int)2*w + 1, signma, true, false);
		Mat L = DoG_pyramid[oct][s];

		vector<float> hist(num_bins, 0);

		int mx_bin = -1;
		float mx_hist = -1e9;
	
		for (int oy = -w; oy <= w; ++oy) {
			for (int ox = -w; ox <= w; ++ox) {
				int cur_x = x + ox, cur_y = y + oy;
				if (cur_x < 0 || cur_x >= L.cols || cur_y < 0 || cur_y >= L.rows) continue;
				
				float dy = getValueOfMatrix(L, min(cur_y + 1, L.rows - 1), cur_x) - getValueOfMatrix(L, max(cur_y - 1, 0), cur_x),
					dx = getValueOfMatrix(L, cur_y, min(cur_x + 1, L.cols - 1)) - getValueOfMatrix(L, cur_y, max(cur_x - 1, 0));
				float magnitude = sqrt(dx*dx + dy*dy), PI = 2 * acos(0);
				float theta = atan2(dy, dx) * 180 / PI;
				if (theta < 0) theta += 360;

				float weight = getValueOfMatrix(kernel, oy + w, ox + w) * magnitude;
				int bin = (int)(floor(theta) / binwidth);

				assert(bin >= 0 && bin < hist.size());

				hist[bin] += abs(weight);
				if (hist[bin] > mx_hist) 
					mx_bin = bin, mx_hist = hist[bin];
			}
		}

		orientation_kpts.push_back(myKeyPoint(point.y_image, point.x_image, point.octave_index, point.layer_index, fit_parabol(hist, mx_bin, binwidth)));

		for (int i=0;i<hist.size();++i){
			if (i == mx_bin) continue;
			if (0.8 * mx_hist < hist[i])
				orientation_kpts.push_back(myKeyPoint(point.y_image, point.x_image, point.octave_index, point.layer_index, fit_parabol(hist, mx_bin, binwidth)));
		}
	}
	return orientation_kpts;
}

float SiftDetector::fit_parabol(const vector<float> &hist, int mx_bin, int binwidth) {
	int centerval, rightval, leftval, n = hist.size();
	centerval = mx_bin * binwidth + binwidth/2;
	rightval = (mx_bin == n - 1) ? (360 + binwidth / 2) : ((mx_bin + 1) * binwidth + binwidth / 2);
	leftval = (mx_bin == 0) ? (0 - binwidth / 2) : ((mx_bin - 1) * binwidth + binwidth / 2);

	Mat A = Mat::zeros(3, 3, CV_32FC1);
	A.at<float>(0, 0) = centerval * centerval, A.at<float>(0, 1) = centerval, A.at<float>(0, 2) = 1;
	A.at<float>(1, 0) = rightval * rightval, A.at<float>(1, 1) = rightval, A.at<float>(1, 2) = 1;
	A.at<float>(2, 0) = leftval * leftval, A.at<float>(2, 1) = leftval, A.at<float>(2, 2) = 1;

	Mat b = Mat::zeros(3, 1, CV_32FC1);
	b.at<float>(0, 0) = hist[mx_bin], 
	b.at<float>(1, 0) = hist[(mx_bin + 1) % n],
	b.at<float>(2, 0) = hist[(mx_bin + n - 1) % n];

	Mat x = A.inv()*b;
	if (abs(x.at<float>(0) - 0.0) < 1e-6) x.at<float>(0) = 1e-4;
	return -x.at<float>(1) / (2 * x.at<float>(0));
}

void SiftDetector::get_local_descriptors(int oct, const vector<vector<Mat>> &DoG_pyramid, vector<vector<myKeyPoint>> &orient_keypoint, int window_size, int num_subregions, int num_bins) {
	int n_layer = DoG_pyramid[oct].size(), binwidth = 360 / num_bins;

	for (myKeyPoint &point : orient_keypoint[oct]) {
		int y = (int)point.y_image, x = (int)point.x_image, s = (int)point.layer_index;
		(s < 0) ? (s = 0) : ((s >= n_layer) ? (s = n_layer - 1) : s);
		
		Mat L = DoG_pyramid[oct][s];

		int t = max(0, y - window_size / 2), b = min(L.rows - 1, y + window_size / 2);
		int l = max(0, x - window_size / 2), r = min(L.cols - 1, x + window_size / 2);

		int size_of_kernel = min(b - t + 1, r - l + 1);
		(size_of_kernel % 2 == 0) ? (++size_of_kernel) : (1);

		Mat kernel = createGaussianKernel(size_of_kernel, window_size / 6.0, true, false);

		int subregion_window = window_size / num_subregions;
		vector<vector<float>> hist(subregion_window * subregion_window, vector<float>(num_bins, 0));

		for (int yy = t; yy <= b; ++yy) {
			for (int xx = l; xx <= r; ++xx) {
				float dx, dy;
				dy = getValueOfMatrix(L, min(yy + 1, L.rows - 1), xx) - getValueOfMatrix(L, max(yy - 1, 0), xx);
				dx = getValueOfMatrix(L, yy, min(xx + 1, L.cols - 1)) - getValueOfMatrix(L, yy, max(xx - 1, 0));

				float magnitude = sqrt(dx*dx + dy*dy), PI = 2 * acos(0);
				float theta = atan2(dy, dx) * 180 / PI;
				if (theta < 0) theta += 360;

				float weight = getValueOfMatrix(kernel, yy - t, xx - l) * magnitude;
				int bin = (int)(floor(theta) / binwidth);

				assert(bin >= 0 && bin < num_bins);

				hist[min((yy - t) / 4, 3) * 4 + min((xx - l) / 4, 3)][bin] += weight;
				
				Mat feature = Mat::zeros(num_bins * num_subregions * num_subregions, 1, CV_32FC1);

				for (int i = 0; i < num_subregions * num_subregions; ++i) {
					for (int j = 0; j < num_bins; ++j) {
						feature.at<float>(i * num_bins + j, 0) += hist[i][j];
						//cout << feature.at<float>(i * num_bins + j, 0) << endl;
					}
				}

				feature /= max(1e-6, norm(feature, NORM_L1));
				for (int i = 0; i < feature.rows; ++i) 
					if (feature.at<float>(i, 0) > 0.2)
						feature.at<float>(i, 0) = 0.2;
				feature /= max(1e-6, norm(feature, NORM_L1));
			
				point.feature = feature;
			}
		}
	}
}

void SiftDetector::writingKeyPointToFile(const string &filename, const vector<vector<myKeyPoint>> &key_points) {
	ofstream outfile;
	outfile.open(filename);
	for (int oct = 0; oct < key_points.size(); ++oct) {
		for (myKeyPoint point : key_points[oct]) {
			outfile << point.octave_index << " " << point.layer_index << " " << point.y_image << " " << point.x_image
				<< " " << point.hist_parabol_extrema << '\n';

			for (int j = 0; j < point.feature.rows; ++j)
				outfile << point.feature.at<float>(j, 0) << " ";
			outfile << '\n';
		}
	}
	cout << "Successful! Your keypoint traits have been written to " << filename << '\n';
	outfile.close();
}

vector<vector<myKeyPoint>> SiftDetector::siftDetector(const Mat &source, bool wait_Key, int num_octaves, int num_scale_signma, float signma, float thresh_edge, float thresh_contrast, int windowSize){
	Mat image = source.clone();
	/* Step 1: Convert Image to GrayScale, Blur and Double the Image */
	Mat srcGray = convertToGrayScale(image);

	filter2D(srcGray, srcGray, -1, createGaussianKernel(5, 1.3));
	
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
	int num_bins = 36;
	vector<vector<myKeyPoint>> keyPoints_orientation(num_octaves);

	for (int oct = 0; oct < num_octaves; ++oct) 
		for (myKeyPoint kpts : orientation_assignment_for_octave(oct, DoG_pyramid, keyPoints, num_bins))
			keyPoints_orientation[oct].push_back(kpts);

	/* Step 7: Get Keypoint Local Descriptor -> return: 128 long feature vector */

	for (int oct = 0; oct < num_octaves; ++oct)
		get_local_descriptors(oct, DoG_pyramid, keyPoints_orientation, windowSize);

	//return keyPoints_orientation;

	/* Some workspace below: */

	/* Step n: Draw the KeyPoints */
	Mat dst = source.clone();
	for (int oct = 0; oct < num_octaves; ++oct) {
		for (myKeyPoint points : keyPoints_orientation[oct])
			if (oct == 1) 
				circle(dst, Point(points.x_image, points.y_image), sqrt(2), Scalar(0, 0, 255), 2, 8, 0);//(y,x) -> Point(x,y)
				//cout << points.x_image << " " << points.y_image << endl;
		cout << "Choose : " << keyPoints_orientation[oct].size() << " points from octave " << oct << endl;
	}
	/* Step n+1: Show the KeyPoints image */
	namedWindow("Sift_Detector");
	imshow("Sift_Detector", dst);
	if (wait_Key) waitKey(0);
	else
		_sleep(5000);

	return keyPoints_orientation;
}