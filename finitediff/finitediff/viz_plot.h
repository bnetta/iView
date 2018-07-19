//=================================
// include guard
#ifndef VIZ_PLOT_INCLUDED_
#define VIZ_PLOT_INCLUDED_

//=================================
// included dependencies
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\viz.hpp>

#include <ppl.h>

//=================================
// the actual class
class viz_plot
{
public:
	void vizPlot3D(const cv::Mat depthmap, const cv::Mat colormap, const float pixel_width_mm, const float pixel_height_mm);
};

#endif 

