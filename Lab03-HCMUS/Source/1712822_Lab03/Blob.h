/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 20.05.2020 16:50:40
**/

#ifndef BLOB_H
#define BLOB_H

#include "Utils.h"

class BlobDetector {
public:
	set<tuple<int, int, float>> detectBlob(const Mat &source, float signma = 1.0, float k = sqrt(2), float thresholdMax = 0.3);
	void showBlobPoint_BlobDetector(const Mat &source, set<tuple<int, int, float>> blobPoints, bool wait_Key = true);

	set<tuple<int, int, float>> detectDOG(const Mat &source, float signma = 1.0, float k = sqrt(2), float thresholdMax = 0.3);
	void showBlobPoint_DOGDetector(const Mat &source, set<tuple<int, int, float>> blobPoints, bool wait_Key = true);
};

#endif