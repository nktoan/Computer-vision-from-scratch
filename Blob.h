/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 20.05.2020 16:50:40
**/

#ifndef BLOB_H
#define BLOB_H

#include "Utils.h"

class BlobDetector {
public:
	void detectBlob(const Mat &source, bool is_show = true, bool wait_Key = true, float signma = 1.0, float k = sqrt(2), float thresholdMax = 0.3);
	void detectDOG(const Mat &source, bool is_show = true, bool wait_Key = true, float signma = 1.0, float k = sqrt(2), float thresholdMax = 0.3);
};

#endif