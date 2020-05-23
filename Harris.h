/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 20.05.2020 15:50:40
**/

#ifndef HARRIS_H
#define HARRIS_H

#include"Utils.h"

class HarrisDetector {
public:
	void detectHarris(const Mat &source, float k = 0.05, float thresh = 0.01);
};

#endif