#include "viz_plot.h"

using namespace std;
using namespace cv;


void viz_plot::vizPlot3D(const Mat depthmap, const Mat colormap, const float pixel_width_mm, const float pixel_height_mm) {

	viz::Viz3d myWindow("Point Cloud");

	float Width = depthmap.cols;
	float Height = depthmap.rows;

	float X, Y, Z;

	std::vector<cv::Vec3f> buffer(Width * Height, cv::Vec3f::all(std::numeric_limits<float>::quiet_NaN()));

	for (int pixel_row = 1; pixel_row < Height; pixel_row++){
		const unsigned int depthOffset = pixel_row * Width;

		for (int pixel_col = 1; pixel_col < Width; pixel_col++){
			unsigned int depthIndex = depthOffset + pixel_col;

			X = pixel_col*pixel_width_mm;
			Y = pixel_row*pixel_height_mm;
			Z = depthmap.at<float>(pixel_row, pixel_col);

			buffer[depthIndex] = cv::Vec3f(X, Y, Z);

		}
	}

	// Create cv::Mat from Coordinate Buffer
	Mat cloudMat = cv::Mat(Height, Width, CV_32FC3, &buffer[0]).clone();


	viz::WCloud cloud(cloudMat, colormap);

	myWindow.showWidget("Cloud", cloud);

	/// Rodrigues vector
	Mat rot_vec = Mat::zeros(1, 3, CV_32F);
	float translation_phase = 0.0, translation = 0.0;
	while (!myWindow.wasStopped())
	{

		myWindow.spinOnce(1, true);
	}
}



