//=================================
// include guard
#ifndef RECONSTRUCTION_INCLUDED_
#define RECONSTRUCTION_INCLUDED_

//=================================
// included dependencies
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>


//=================================
// the actual class
class reconstruction
{
public:
	void reconCentroid(const std::vector<cv::Mat> AHO, cv::Mat depthmap, cv::Mat colormap);
};

#endif 

