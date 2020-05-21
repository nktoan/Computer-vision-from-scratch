/**
*    author:  Khanh-Toan Nguyen, 1712822, 17TN, HCMUS.
*    created: 20.05.2020 15:50:40
**/

#ifndef ImageOperator_H
#define ImageOperator_H

#include"Utils.h"

Mat convertToGrayScale(const Mat &source);

void detectHarris(const Mat &source, float k = 0.05, float thresh = 0.01);

#endif