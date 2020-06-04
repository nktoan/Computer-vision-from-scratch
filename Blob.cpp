/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 20.05.2020 16:50:50
**/

#include "Blob.h"

set<tuple<int, int, float>> BlobDetector::detectBlob(const Mat &source, float signma, float k, float thresholdMax) {
	/* Step 1: Convert Image to GrayScale */
	Mat srcGray = convertToGrayScale(source);

	/* Step 2: Convolution srcGray with LoG filter at 10 scales
	output store in vector<Mat> log_image (size = 10)
	each elements of log_image represents the output of convolution image with corresponding scale */
	int number_of_scales = 10;
	vector<Mat> log_image(number_of_scales, Mat::zeros(source.size(), CV_32FC1));
	float signma_y = signma;

	vector<float> max_log(number_of_scales, 0);

	for (int idx = 0; idx < log_image.size(); ++idx) {
		signma_y = (idx == 0) ? signma_y : (signma_y * k);

		Mat log_filter = createLoG_Kernel(5, signma_y, true, true);
		Mat conv_result;

		filter2D(srcGray, conv_result, CV_32FC1, log_filter);

		conv_result = multiplyElementWise(conv_result, conv_result);

		max_log[idx] = getMaxValueOfMatrix(conv_result);

		log_image[idx] = conv_result;
	}

	/* Step 3: Finding the maximum peak, comparing with 26 points */
	set<tuple<int, int, float>> blob_points;
	for (int idx = 0; idx < log_image.size(); ++idx) {
		for (int y = 0; y < source.rows; ++y) {
			for (int x = 0; x < source.cols; ++x) {
				float val = getValueOfMatrix(log_image[idx], y, x);
				if (val <= thresholdMax * max_log[idx]) continue;

				bool found_peak = true;
				for (int step_idx = -1; step_idx <= 1; ++step_idx) {
					if (found_peak == false) break;

					for (int step_x = -1; step_x <= 1; ++step_x) {
						if (found_peak == false) break;

						for (int step_y = -1; step_y <= 1; ++step_y) {
							if (found_peak == false) break;

							int cur_idx = idx + step_idx, cur_y = y + step_y, cur_x = x + step_x;
							if (cur_idx >= log_image.size() || cur_idx < 0) continue;
							if (cur_y >= source.rows || cur_y < 0) continue;
							if (cur_x >= source.cols || cur_x < 0) continue;

							if (val < getValueOfMatrix(log_image[cur_idx], cur_y, cur_x))
								found_peak = false;
						}
					}
				}

				if (found_peak == true)
					blob_points.insert(make_tuple(y, x, pow(k, idx)*signma));
			}
		}
	}

	/* Step 4: NMS */
	
	return blob_points;
}

void BlobDetector::showBlobPoint_BlobDetector(const Mat &source, set<tuple<int, int, float>> blob_points, bool wait_Key){
	/* Draw the blobs */
	Mat dst = source.clone();
	for (tuple<int, int, float> point : blob_points)
		circle(dst, Point(get<1>(point), get<0>(point)), get<2>(point) * sqrt(2), Scalar(0, 0, 255), 2, 8, 0);//(y,x) -> Point(x,y)

	 /* Show the blob image */
	namedWindow("Blob_detector");
	imshow("Blob_detector", dst);
	if (wait_Key) waitKey(0);
	else
		_sleep(5000);
}
set<tuple<int, int, float>> BlobDetector::detectDOG(const Mat &source, float signma, float k, float thresholdMax) {
	/* Step 1: Convert Image to GrayScale */
	Mat srcGray = convertToGrayScale(source);

	/* Step 2: Computer DoG_filter (vector<Mat> dog(10, G(x,y,k*signma)-G(x,y,signma)) */
	int number_of_scales = 10;
	vector<Mat> DoG_filter(number_of_scales, Mat::zeros(source.size(), CV_32FC1));

	for (int i = 0; i < number_of_scales; ++i)
		DoG_filter[i] = createGaussianKernel(5, pow(k, i)*signma, false, true);

	/* Step 3: Convolution with DoG */
	vector<Mat> DoG_conv(number_of_scales - 1, Mat::zeros(source.size(), CV_32FC1));
	vector<float> max_DoG(number_of_scales, 0);

	for (int i = 0; i < DoG_conv.size(); ++i) {
		Mat conv_result_i, conv_result_i_plus_1;

		filter2D(srcGray, conv_result_i, CV_32FC1, DoG_filter[i]);
		filter2D(srcGray, conv_result_i_plus_1, CV_32FC1, DoG_filter[i + 1]);

		Mat conv_result = mimusElementWise(conv_result_i_plus_1, conv_result_i);

		conv_result = multiplyElementWise(conv_result, conv_result);

		max_DoG[i] = getMaxValueOfMatrix(conv_result);
		DoG_conv[i] = conv_result;
	}

	/* Step 4: Finding the maximum peak, comparing with 26 points */
	set<tuple<int, int, float>> blob_points;
	for (int idx = 0; idx < DoG_conv.size(); ++idx) {
		for (int y = 0; y < source.rows; ++y) {
			for (int x = 0; x < source.cols; ++x) {
				float val = getValueOfMatrix(DoG_conv[idx], y, x);
				if (val <= thresholdMax * max_DoG[idx]) continue;

				bool found_peak = true;
				for (int step_idx = -1; step_idx <= 1; ++step_idx) {
					if (found_peak == false) break;

					for (int step_x = -1; step_x <= 1; ++step_x) {
						if (found_peak == false) break;

						for (int step_y = -1; step_y <= 1; ++step_y) {
							if (found_peak == false) break;

							int cur_idx = idx + step_idx, cur_y = y + step_y, cur_x = x + step_x;
							if (cur_idx >= DoG_conv.size() || cur_idx < 0) continue;
							if (cur_y >= source.rows || cur_y < 0) continue;
							if (cur_x >= source.cols || cur_x < 0) continue;

							if (val < getValueOfMatrix(DoG_conv[cur_idx], cur_y, cur_x))
								found_peak = false;
						}
					}
				}

				if (found_peak == true)
					blob_points.insert(make_tuple(y, x, pow(k, idx)*signma));
			}
		}
	}

	/* Step 4: NMS */

	return blob_points;
}
void BlobDetector::showBlobPoint_DOGDetector(const Mat &source, set<tuple<int, int, float>> blob_points, bool wait_Key) {
	/* Draw the blobs */
	Mat dst = source.clone();
	for (tuple<int, int, float> point : blob_points)
		circle(dst, Point(get<1>(point), get<0>(point)), get<2>(point) * sqrt(2), Scalar(0, 0, 255), 2, 8, 0);//(y,x) -> Point(x,y)

	/* Show the blob image */
	namedWindow("DOG_detector");
	imshow("DOG_detector", dst);
	if (wait_Key) waitKey(0);
	else
		_sleep(5000);
}