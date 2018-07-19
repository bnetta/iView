//=================================
// include guard
#ifndef VECTOR_TOOLS_INCLUDED_
#define VECTOR_TOOLS_INCLUDED_

//=================================
// included dependencies
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\viz.hpp>

#include <ppl.h>

#include <numeric>

//=================================

std::vector<float> extractFocusCurve(std::vector<cv::Mat>focus_volume, int row_index, int col_index);

float vectorMax(std::vector<float>vector);

void calcCSTD(std::vector<float> fm_vector, std::vector<float> z_pos_vector, std::vector<float>& fm_vector_norm, float z_step, float decay_rate, float& curve_stdev, float& alpha);


#endif 