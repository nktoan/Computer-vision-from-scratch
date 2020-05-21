/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 20.05.2020 15:50:40
**/

#ifndef ImageOperator_H
#define ImageOperator_H

#include"Utils.h"

Mat convertToGrayScale(const Mat &source);

void detectHarris(const Mat &source, float k = 0.05, float thresh = 0.01);
void detectBlob(const Mat &source, float signma = 1.0, float k = sqrt(2), float thresholdMax = 0.3);
void detectDOG(const Mat &source, float signma = 1.0, float k = sqrt(2), float thresholdMax = 0.3);
void sift(const Mat &source);

#endif