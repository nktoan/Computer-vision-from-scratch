/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 20.05.2020 15:50:50
**/

#include "Harris.h"

void HarrisDetector::detectHarris(const Mat &source, float k, float thresh) {
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

	/* Step 4: compute (Gx)^2, (Gy)^2, Gx.Gy and then apply convolution with window function (gaussian function) */

	Mat gradient_x_square = multiplyElementWise(gradient_x, gradient_x);
	Mat gradient_y_square = multiplyElementWise(gradient_x, gradient_y);
	Mat gradient_x_y = multiplyElementWise(gradient_y, gradient_y);

	filter2D(gradient_x_square, gradient_x_square, CV_32FC1, createGaussianKernel(3, 1));
	filter2D(gradient_y_square, gradient_y_square, CV_32FC1, createGaussianKernel(3, 1));
	filter2D(gradient_x_y, gradient_x_y, CV_32FC1, createGaussianKernel(3, 1));

	/* Step 5: create a matrix 2x2 M at each position (y, x) = [[gradient_x_square, gradient_x_y], [gradient_x_y, gradient_y_square]]
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
	vector<CornerPoint> corner_points;

	for (int y = 0; y < R.rows; ++y) {
		for (int x = 0; x < R.cols; ++x) {
			float r_val = getValueOfMatrix(R, y, x);
			if (r_val > thresh * r_max)
				corner_points.push_back(CornerPoint(r_val, y, x));
		}
	}
	sort(corner_points.begin(), corner_points.end());
	reverse(corner_points.begin(), corner_points.end());

	//6.2
	int distance = 10;
	vector<CornerPoint> nms_corner_points;

	for (CornerPoint point_1 : corner_points) {
		if (nms_corner_points.size() > 0) {
			bool not_found = true;
			for (CornerPoint point_2 : nms_corner_points)
				not_found &= (abs(point_1.x - point_2.x) >= distance) || (abs(point_1.y - point_2.y) >= distance);
			
			if (not_found == true) 
				nms_corner_points.push_back(point_1);
		}
		else 
			nms_corner_points.push_back(point_1);
	}

	/* Step 7: Draw the corners */
	
	Mat dst = source.clone();
	for (CornerPoint point : nms_corner_points)
		circle(dst, Point(point.x, point.y), 4, Scalar(0, 0, 255), 2, 8, 0);//(y,x) -> Point(x,y)

	/* Step 8: Show corner image */
	namedWindow("cornersDetector_Harris");
	imshow("cornersDetector_Harris", dst);
	waitKey(0);

	/* Step 9: Store result (Optional) */

	//printMatrixInfo(dst);
}

