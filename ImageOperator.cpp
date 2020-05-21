/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 20.05.2020 15:50:50
**/

#include "ImageOperator.h"

Mat convertToGrayScale(const Mat &source) {
	Mat dst;
	cvtColor(source, dst, COLOR_BGR2GRAY);
	return dst;
}

void detectHarris(const Mat &source, float k, float thresh) {
	/* Step 1: Convert Image to GrayScale */
	Mat srcGray = convertToGrayScale(source);

	/* Step 2: Blur the image to reduce noise */
	Mat srcGrayBlur, gaussianKernel5x5 = createGaussianKernel();
	filter2D(srcGray, srcGrayBlur, -1, gaussianKernel5x5);
	
	/* Step 3: compute gradient Gx, Gy */
	Mat gradient_x, gradient_y;
	Mat sobel_x = createSobelX(), sobel_y = createSobelY();

	filter2D(srcGrayBlur, gradient_x, CV_32FC1, sobel_x);
	filter2D(srcGrayBlur, gradient_y, CV_32FC1, sobel_y);

	/* Step 4: compute (Gx)^2, (Gy)^2, Gx.Gy */

	Mat gradient_x_square = multiplyElementWise(gradient_x, gradient_x);
	Mat gradient_y_square = multiplyElementWise(gradient_x, gradient_y);
	Mat gradient_x_y = multiplyElementWise(gradient_y, gradient_y);

	/* Step 5: create a matrix 2x2 M at each position (y, x) = [[gradient_x_square, gradient_x_y], 
																[gradient_x_y, gradient_y_square]]
				then we have R[y, x] = det(M) - k. (trace(M))^2 */
	float r_max = -1e9;
	Mat R = Mat::zeros(source.size(), CV_32FC1);

	for (int y = 0; y < R.rows; ++y) {
		for (int x = 0; x < R.cols; ++x) {
			/* create matrix 2x2 M */
			Mat M = Mat::zeros(2, 2, CV_32FC1);
			setValueOfMatrix(M, 0, 0, getValueOfMatrix(gradient_x_square, y, x));
			setValueOfMatrix(M, 0, 1, getValueOfMatrix(gradient_x_y, y, x));
			setValueOfMatrix(M, 1, 0, getValueOfMatrix(gradient_x_y, y, x));
			setValueOfMatrix(M, 1, 1, getValueOfMatrix(gradient_y_square, y, x));
			
			/* compute det(M) and trace(M) */
			float det_M = getValueOfMatrix(M, 0, 0) * getValueOfMatrix(M, 1, 1) 
										- getValueOfMatrix(M, 1, 0) * getValueOfMatrix(M, 0, 1);

			float trace_M = getValueOfMatrix(M, 0, 0) + getValueOfMatrix(M, 1, 1);

			float r_val = det_M - k * trace_M * trace_M;
			r_max = max(r_max, r_val);

			setValueOfMatrix(R, y, x, r_val);
		}
	}

	/* Step 6: 6.1. Compare value of R with threshold 
			   6.2. Non-maximum Suppression to suppress some consecutive corner-point within Distance range.
			   -> output set<pair<int,int>>: nms_corner_points */
	//6.1
	set<pair<int, int>> corner_points;

	for (int y = 0; y < R.rows; ++y) {
		for (int x = 0; x < R.cols; ++x) {
			float r_val = getValueOfMatrix(R, y, x);
			if (r_val > thresh * r_max)
				corner_points.insert(make_pair(y, x));
		}
	}
	//6.2
	int distance = 10;
	set<pair<int, int>> nms_corner_points;

	for (pair<int, int> point_1 : corner_points) {
		if (nms_corner_points.size() > 0) {
			bool not_found = true;
			for (pair<int, int> point_2 : nms_corner_points) 
				not_found &= (abs(point_1.first - point_2.first) >= distance) || (abs(point_1.second - point_2.second) >= distance);
			
			if (not_found == true) 
				nms_corner_points.insert(point_1);
		}
		else 
			nms_corner_points.insert(point_1);
	}

	/* Step 7: Draw the corners */
	
	Mat dst = source.clone();
	for (pair<int, int> point : nms_corner_points)
		circle(dst, Point(point.second, point.first), 4, (0, 0, 255), 2, 8, 0);//(y,x) -> Point(x,y)

	/* Step 8: Show corner image */
	namedWindow("cornersDetector_Harris");
	imshow("cornersDetector_Harris", dst);
	waitKey(0);

	/* Step 9: Store result (Optional) */

	//printMatrixInfo(dst);
}

void detectBlob(const Mat &source, float signma, float k){
	/* Step 1: Convert Image to GrayScale */
	Mat srcGray = convertToGrayScale(source);

	/* Step 2: Convolution srcGray with LoG filter at 10 scales
				output store in vector<Mat> log_image (size = 10)
					each elements of log_image represents the output of convolution image with corresponding scale*/

	vector<Mat> log_image(10, Mat::zeros(source.size(), CV_32FC1));
	float signma_y = signma;

	for (int index = 0; index < log_image.size(); ++index) {
		signma_y = (index == 0) ? signma_y : (signma_y * k);

		Mat log_filter = createLoG_Kernel(5, signma_y);
		Mat conv_result;
		filter2D(srcGray, conv_result, CV_32FC1, log_filter);
		conv_result = multiplyElementWise(conv_result, conv_result);

		log_image[index] = conv_result;
	}

	/* Step 3: Finding the maximum peak, comparing with 26 points */
	set<tuple<int, int, float>> blob_points;

	for (int y = 0; y < source.rows; ++y) {
		for (int x = 0; x < source.cols; ++x) {
			for (int index = 0; index < log_image.size(); ++index) {
				float val = getValueOfMatrix(log_image[index], y, x);

				bool found_peak = true;
				for (int step_idx = -10; step_idx <= 10; ++step_idx) {
					if (found_peak == false) break;

					for (int step_x = -1; step_x <= 1; ++step_x) {
						if (found_peak == false) break;

						for (int step_y = -1; step_y <= 1; ++step_y) {
							if (found_peak == false) break;

							int cur_idx = index + step_idx, cur_y = y + step_y, cur_x = x + step_x;
							if (cur_idx >= log_image.size() || cur_idx < 0) continue;
							if (cur_y >= source.rows || cur_y < 0) continue;
							if (cur_x >= source.cols || cur_x < 0) continue;

							if (val < getValueOfMatrix(log_image[cur_idx], cur_y, cur_x))
								found_peak = false;
						}
					}
				}

				if (found_peak == true && val > 0.03)
					blob_points.insert(make_tuple(y, x, pow(k, index)*signma));
			}
		}
	}

	/* Step 4: NMS */

	/* Step 5: Draw the blobs */
	Mat dst = source.clone();
	for (tuple<int, int, float> point : blob_points)
		circle(dst, Point(get<1>(point), get<0>(point)), get<2>(point)*sqrt(2), (0, 0, 255), 2, 8, 0);//(y,x) -> Point(x,y)

	/* Step 6: Show the blob image */
	namedWindow("Blob_detector");
	imshow("Blob_detector", dst);
	waitKey(0);

}
void detectDOG(const Mat &source, float signma, float k){


}
