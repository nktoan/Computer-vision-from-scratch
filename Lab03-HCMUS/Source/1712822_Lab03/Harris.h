/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 20.05.2020 15:50:40
**/

#ifndef HARRIS_H
#define HARRIS_H

#include"Utils.h"

class CornerPoint {
public:
	float r_value;
	int x, y;
public:
	CornerPoint(float r_val, int y_val, int x_val) : r_value(r_val), y(y_val), x(x_val) {};

	bool operator < (const CornerPoint &other) {
		return r_value < other.r_value;
	}
};
class HarrisDetector {
public:
	vector<CornerPoint> detectHarris(const Mat &source, float k = 0.05, float alpha = 0.01, float d = 10);
	void showCornerPoint(const Mat& source, const vector<CornerPoint> &cornerPoints, bool wait_Key = true);
};

#endif