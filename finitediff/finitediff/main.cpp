#define _CRT_SECURE_NO_DEPRECATE
#include "finitediff\include\finitediff_templated.hpp"
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>

#include <opencv2\plot.hpp>

#include "viz_plot.h"
#include "vector_tools.h"

#include <vector>
#include <string>
#include <numeric>

#include <ppl.h>

#include "lmmin.h"
#include "lmcurve.h"

#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;
using namespace cv;
using namespace finitediff;
using namespace concurrency;

const bool GENERATE_AHO = true;	//true = generate the focus volume from images | false = use pre-generated
const bool DO_AGGRIGATION = true;

cv::Point2i pt(-1, -1); //define a point for use in mouse selection

//data set information
const float z_step = 0.1;				//the step between each image
const float pixel_width_mm = 0.004;
const float pixel_height_mm = 0.004;

//hyperparameters
float decay_rate = 6;			//6 specified in paper
int max_derivative_order = 2;	//number of derivatives to run
int kernel_size = 3;			//must be odd and greater than derivative order
int agg_iterations = 6;			//aggrigation iterations
int agg_filter_size = 5;		//filter size used in aggrigation
float rec_threshold = 0.7;		//threshold used in reconstruction. Defines what fraction of points are to be considered. 
float rec_alpha_threshold = 0.9;	//only add values with this confidence to depthmap

//open all the images
String folderpath = "C:/git_repos/iView/simu02/*.tif";
//String folderpath = "C:/Users/jacobge/Desktop/GNU Octave/sff/iview scan/*.bmp";
//String folderpath = "C:/Users/jacobge/Desktop/GNU Octave/sff/Alicona aligned/*.jpg";
//string folderpath = "C:/Users/jacobge/Desktop/Alicona/*.bmp";

//some global variables to be used by mouse click curve inspection
vector<Mat> AHO;	// adaptive higher order focus volume
vector<float> z_pos_vector; //the vector of image stack z positions

Mat fnorm_map;
Mat fit_stdv_map;
Mat alpha_mat;
Mat curve_stdev_mat;

Mat genKernel(int kernel_size, int der_order, int direction){

	//generate the weights for the image filter
	const unsigned max_deriv = der_order;
	
	std::vector<double> x; //positions to asses derivative i.e. pixel locations
	//std::vector<double> x{ -4, -3, -2, -1, 0, 1, 2, 3, 4 };  // Fourth order of accuracy
	float pixel_pos;
	for (int i = 0; i < kernel_size; i++){
		pixel_pos = i - ((kernel_size - 1) / 2);
		if ((direction == 3) | (direction == 4)){
			pixel_pos = pixel_pos / sqrt(2); //pixels are furthur apart in diagonal directions
		}
		x.push_back(pixel_pos);
	}
	
	
	auto coeffs = finitediff::generate_weights(x, max_deriv);

	//generate image kernels of various derivitive orders
	std::vector<double> kernel; //a vector to store the kernel values

	if ((direction == 1) | (direction == 2)){ //pad out vector for x and y directions
		for (unsigned row_idx = 0; row_idx < x.size(); row_idx++) {
			if (row_idx == ((x.size() - 1) / 2)){
				for (unsigned idx = 0; idx < x.size(); idx++) {
					kernel.push_back(coeffs[der_order*x.size() + idx]);
				}
			}
			else{
				for (unsigned idx = 0; idx < x.size(); idx++) {
					kernel.push_back(0);
				}
			}
		}
	}
	else if ((direction == 3) | (direction == 4)){ //pad out vector diagonal directions
		for (unsigned row_idx = 0; row_idx < x.size(); row_idx++) {
			for (unsigned col_idx = 0; col_idx < x.size(); col_idx++) {
				if (row_idx == col_idx){
					kernel.push_back(coeffs[der_order*x.size() + row_idx]);
				}
				else{
					kernel.push_back(0);
				}
			}
		}
	}

	Mat kernel_mat = Mat(kernel, true); //copy the weights vector to a cv Mat
	kernel_mat = kernel_mat.reshape(1, kernel_size);

	if ((direction == 2) | (direction == 4)){
		cv::rotate(kernel_mat, kernel_mat, ROTATE_90_CLOCKWISE);
	}

	return kernel_mat;
}

void normaliseAndDisplay(std::vector<Mat>focus_measure_stack, int image_no, string window_name){
	double min, max;
	cv::minMaxLoc(focus_measure_stack[image_no], &min, &max);
	cout << "Max = " << max << " Min = " << min << " image_no = " << image_no << endl;

	//normalize the filtered image for display
	Mat image_focus_measure_norm;
	cv::normalize(focus_measure_stack[image_no], image_focus_measure_norm, 1.0, 0.0, NORM_MINMAX);

	cv::imshow(window_name, image_focus_measure_norm);
	cv::minMaxLoc(image_focus_measure_norm, &min, &max);
	cout << "Max = " << max << " Min = " << min << " image_no = " << image_no << endl;

	cv::waitKey(0);

 }

//inspect a focus curve by plottting
void inspectFocusCurve(std::vector<float>fm_vector, std::vector<float>z_vector, string title){
		
	Mat data_x(1, z_vector.size(), CV_64F);
	Mat data_y(1, fm_vector.size(), CV_64F);

	for (int i = 0; i < fm_vector.size(); i++){
		data_x.at<double>(0, i) = fm_vector[i];
		data_y.at<double>(0, i) = z_vector[i];
	}

	Mat plot_result;

	Ptr<plot::Plot2d> plot = plot::Plot2d::create(data_x, data_y);
	plot->render(plot_result);

	imshow(title, plot_result);

	waitKey(0);

	destroyAllWindows();
	
}


// model of gaussian function to use when curve fitting
double f(const double t, const double *p)
{
	return (p[0] / (p[1] * sqrt(2 * M_PI)))*exp(-((t - p[2])*(t - p[2])) / (2 * p[1] * p[1]));
	
}


//get position of mouse click
void onMouse(int evt, int x, int y, int flags, void* param) {
	if (evt == CV_EVENT_LBUTTONDOWN) {
		cv::Point* ptPtr = (cv::Point*)param;
		ptPtr->x = x;
		ptPtr->y = y;
		cout << x << " ;" << y << "; curve_stdev = " << curve_stdev_mat.at<float>(y, x) << "; alpha = " << alpha_mat.at<float>(y, x) << endl;  // << " fnorm = " << fnorm_map.at<float>(y, x) << " stdv = " << fit_stdv_map.at<float>(y, x) 

		vector<float> focus_curve = extractFocusCurve(AHO, y, x); //row-major order
		for (int i = 0; i < focus_curve.size(); i++){
			cout << focus_curve[i] << ";";
		}
		cout << endl;


		inspectFocusCurve(focus_curve, z_pos_vector, "Selected Focus Curve");
		

	}
}

void reconstruction(vector<Mat> AHO, vector<Mat> input_image_stack, vector<float> z_pos_vector, Mat& depthmap, Mat& colormap){
	//************  Reconstruction  *********************
	//get the z heights from focus curve centroid
	//for each x y pixel find the focus curve centroid
	cout << "Running Reconstruction" << endl;

	int num_rows = input_image_stack[0].rows;
	int num_cols = input_image_stack[0].cols;

	parallel_for(0, num_rows, [&, num_cols](int pixel_row){
		for (int pixel_col = 0; pixel_col < num_cols; pixel_col++)
		{
			//extract the focus curve
			vector<float> rec_fm_vec;
			vector<float> rec_fm_vec_norm;
			rec_fm_vec = extractFocusCurve(AHO, pixel_row, pixel_col);

			float curve_stdev, alpha;

			//calculate the curve standard deviation (CSTD) and confidence measure coefficient alpha
			calcCSTD(rec_fm_vec, z_pos_vector, rec_fm_vec_norm, z_step, decay_rate, curve_stdev, alpha);

			//save the confidence values to a mat for inspection - overwriting old mat - this is the confidence mat post aggrigation
			alpha_mat.at<float>(pixel_row, pixel_col) = alpha;
			curve_stdev_mat.at<float>(pixel_row, pixel_col) = curve_stdev;


			//threshold to only get peak of curve
			vector<float> top_fm_vector;
			vector<float> top_fm_z_pos_vector;
			float z_pos_fm_max;

			for (int image_no = 0; image_no < rec_fm_vec_norm.size(); image_no++){

				if (rec_fm_vec_norm[image_no]>(rec_threshold)){
					top_fm_vector.push_back(rec_fm_vec_norm[image_no]);
					top_fm_z_pos_vector.push_back(z_pos_vector[image_no]);

				}
			}

			float sum_top = std::accumulate(top_fm_vector.begin(), top_fm_vector.end(), 0.0);

			//calculate the centroid of rotation about the fm axis
			float z_centroid = 0;
			float inertia_sum = 0;
			for (int i_z = 0; i_z < top_fm_vector.size(); i_z++){
				inertia_sum += (top_fm_vector[i_z] * z_step) * top_fm_z_pos_vector[i_z];
			}

			z_centroid = inertia_sum / (sum_top * z_step);

			//decide whether to add data to depthmap
			if (alpha >= rec_alpha_threshold) {

				depthmap.at<float>(pixel_row, pixel_col) = z_centroid;

				//cout << "z_centroid = " << z_centroid << endl;

				//retrieve the pixel value for this z height from the image stack and add to focused colormap -- TODO: add interpolation here
				float focus_cut_float = std::round(z_centroid / z_step);
				int focus_cut_int = focus_cut_float - 1;
				if (focus_cut_int >= input_image_stack.size()){
					focus_cut_int = input_image_stack.size() - 1;
				}
				if (focus_cut_int < 0){
					focus_cut_int = 0;
				}

				colormap.at<unsigned char>(pixel_row, pixel_col) = input_image_stack[focus_cut_int].at<unsigned char>(pixel_row, pixel_col);

			}
			else{
				depthmap.at<float>(pixel_row, pixel_col) = NAN;	//set to NAN so these pixels aren't displayed 
				colormap.at<unsigned char>(pixel_row, pixel_col) = NAN;
			}


			rec_fm_vec.clear();
			rec_fm_vec_norm.clear();
			top_fm_vector.clear();
			top_fm_z_pos_vector.clear();

		}
	}); //reconstruction loop
}

void main(){


	vector<String> filenames;

	Mat input_image;

	vector<Mat> input_image_stack;
	vector<Mat> focus_measure_stack;

	Mat AHO_frame;

	//vector<Mat> AHO;	// adaptive higher order focus volume
	//vector<float> z_pos_vector; //the vector of image stack z positions

	vector<Mat> AHO_working;



	glob(folderpath, filenames);

	float z_pos_min = 0;
	float z_pos_max = filenames.size()*z_step;
	float z_pos = z_pos_min;

	for (unsigned int image_no = 0; image_no < filenames.size(); image_no++){

		input_image = imread(filenames[image_no], CV_LOAD_IMAGE_GRAYSCALE);
		if (input_image.empty())
		{
			std::cout << "Image Not Found: " << filenames[image_no] << std::endl;
			return;
		}
		input_image_stack.push_back(input_image);

		//initialise the AHO volume with zeros
		AHO_frame = Mat::zeros(input_image.rows, input_image.cols, CV_32F);
		AHO.push_back(AHO_frame.clone());
		AHO_working.push_back(AHO_frame.clone());

		z_pos_vector.push_back(z_pos);
		z_pos = z_pos + z_step;

	}
	
	int num_rows = input_image.rows;
	int num_cols = input_image.cols;

	alpha_mat = Mat::zeros(num_rows, num_cols, CV_32F);
	curve_stdev_mat = Mat::zeros(num_rows, num_cols, CV_32F);
	Mat depthmap = Mat::zeros(num_rows, num_cols, CV_32F);
	Mat colormap = Mat::zeros(num_rows, num_cols, CV_8UC1);

	fnorm_map = Mat::zeros(num_rows, num_cols, CV_32F);
	fit_stdv_map = Mat::zeros(num_rows, num_cols, CV_32F);


	std::clock_t start;
	double duration;
	start = std::clock();

	colormap.at<unsigned char>(0, 0) = input_image_stack[0].at<unsigned char>(0, 0);
	unsigned char map_pix = input_image_stack[0].at<unsigned char>(0, 0);

	//generate the kernels for image derivative aproximation
	Mat	kernel_mat;
	Mat input_gray;
	Mat image_focus_measure;
	vector<float> fm_vector, fm_vector_norm;

	float curve_stdev, alpha;

	if (GENERATE_AHO){
		//for each derivative order (n) 1-10
		for (int derivative_order = 1; derivative_order <= max_derivative_order; derivative_order++){
			cout << "applying derivative_order = " << derivative_order << endl;

			//for each kernel_direction (m) 1-4
			for (int kernel_direction = 1; kernel_direction <= 4; kernel_direction++){

				focus_measure_stack.clear();

				//generate the kernel
				kernel_mat = genKernel(kernel_size, derivative_order, kernel_direction);

				//generate the focus measure for each image in the stack and add to volume
				for (int image_no = 0; image_no < input_image_stack.size(); image_no++){
					input_gray = input_image_stack[image_no];

					cv::filter2D(input_gray, image_focus_measure, CV_32F, kernel_mat);
					image_focus_measure = cv::abs(image_focus_measure);

					//add filtered image to volume
					focus_measure_stack.push_back(image_focus_measure.clone());
				}

				alpha = 0;
				curve_stdev = 0;


				//for each x-y pixel in the volume
				parallel_for(0, num_rows, [&, num_cols](int pixel_row){
					for (int pixel_col = 0; pixel_col < num_cols; pixel_col++)
					{
						//extract the vector of focus measures from the volume
						vector<float> fm_vec, fm_vec_norm;
						fm_vec = extractFocusCurve(focus_measure_stack, pixel_row, pixel_col);

						//calculate the curve standard deviation (CSTD) and confidence measure coefficient alpha
						calcCSTD(fm_vec, z_pos_vector, fm_vec_norm, z_step, decay_rate, curve_stdev, alpha);

						//save the confidence values to a mat for inspection
						//alpha_mat.at<float>(pixel_row, pixel_col) += alpha;
						//curve_stdev_mat.at<float>(pixel_row, pixel_col) = curve_stdev;

						//add the weighted focus measure to the Adaptive High Order (AHO) focus measure volume.

						//OpenCV, like may other libraries, treats matrices (and images) in row-major order. That means every access is defined as (ROW, column)

						// for each x, y position multiply the volume by alpha

						for (int image_no = 0; image_no < input_image_stack.size(); image_no++){
							float pixel = AHO[image_no].at<float>(pixel_row, pixel_col) + alpha*fm_vec_norm[image_no];
							AHO_working[image_no].at<float>(pixel_row, pixel_col) = pixel;
							if (isnan(pixel)) {
								cout << "pixel " << pixel_row << " " << pixel_col << " focus measure was NAN. Image # " << image_no << "\n" << endl;
							}

							//cout << "added " << alpha*fm_vector_norm[image_no] << " to pixel = " << pixel_row << " " << pixel_col << " image_no " << image_no << " pixel now = " << AHO[image_no].at<float>(pixel_row, pixel_col) << endl;
						}

						fm_vec.clear();
						fm_vec_norm.clear();

					}
				});



				for (int image_no = 0; image_no < input_image_stack.size(); image_no++){
					AHO[image_no] = AHO_working[image_no] + AHO[image_no];
					//normaliseAndDisplay(AHO, image_no, "AHO");
				}
				cout << "|";
				/*if (kernel_direction == 1){
					Mat curve_stdev_mat_dir1;
					namedWindow("curve_stdev_mat_dir1", 0); // Can be resized
					normalize(curve_stdev_mat, curve_stdev_mat_dir1, 1.0, 0.0, NORM_MINMAX);
					cv::setMouseCallback("curve_stdev_mat_dir1", onMouse, (void*)&pt);

					for (;;)
					{
						cv::imshow("curve_stdev_mat_dir1", curve_stdev_mat_dir1);

						char c = (char)waitKey(0);

						if (c == 27)
						{
							cout << "Exiting ...\n";
							break;
						}
					}
				}
				if (kernel_direction == 2){
					Mat alpha_mat_dir2 = curve_stdev_mat;
					//normalise the new alpha_mat and display
					Mat alpha_mat_dir2_norm;
					normalize(alpha_mat_dir2, alpha_mat_dir2_norm, 1.0, 0.0, NORM_MINMAX);
					imshow("alpha_mat_dir2", alpha_mat_dir2);
					waitKey(0);
					alpha_mat = Mat::zeros(num_rows, num_cols, CV_32F);
				}
				if (kernel_direction == 3){
					Mat alpha_mat_dir3 = curve_stdev_mat;
					//normalise the new alpha_mat and display
					Mat alpha_mat_dir3_norm;
					normalize(alpha_mat_dir3, alpha_mat_dir3_norm, 1.0, 0.0, NORM_MINMAX);
					imshow("alpha_mat_dir3", alpha_mat_dir3);
					waitKey(0);
					alpha_mat = Mat::zeros(num_rows, num_cols, CV_32F);
				}
				if (kernel_direction == 4){
					Mat alpha_mat_dir4 = curve_stdev_mat;
					//normalise the new alpha_mat and display
					Mat alpha_mat_dir4_norm;
					normalize(alpha_mat_dir4, alpha_mat_dir4_norm, 1.0, 0.0, NORM_MINMAX);
					imshow("alpha_mat_dir4", alpha_mat_dir4);
					waitKey(0);
					alpha_mat = Mat::zeros(num_rows, num_cols, CV_32F);
				}*/
			}//kernel direction loop



			cout << endl;
			if (derivative_order % 2 != 0){
				kernel_size = kernel_size + 2;
				cout << "kernel_size = " << kernel_size << endl;
			}
		} //derivative order loop

		//save the outputs of the image filtering step
		string filter_filename = "filter_result.yml";
		FileStorage fs1(filter_filename, FileStorage::WRITE);
		fs1 << "AHO" << AHO;
		//fs1 << "alpha_mat" << alpha_mat;
		fs1.release();
		AHO_working.clear();
	}

	else if (DO_AGGRIGATION){
		//if we haven't run the filtering read the data from disk
		string AHO_filename = "filter_result.yml";
		FileStorage fs2(AHO_filename, FileStorage::READ);
		fs2["AHO"] >> AHO;
		//fs2["alpha_mat"] >> alpha_mat;
	


		//show the normalised alpha_mat and display a focus curve based on pixel selection

		namedWindow("alpha_norm", 0); // Can be resized

		Mat alpha_norm;
		cv::normalize(alpha_mat, alpha_norm, 1.0, 0.0, NORM_MINMAX);

		cv::setMouseCallback("alpha_norm", onMouse, (void*)&pt);

		for (;;)
		{
			cv::imshow("alpha_norm", alpha_norm);

			char c = (char)waitKey(0);

			if (c == 27)
			{
				cout << "Exiting ...\n";
				break;
			}
		}

	}

	//************  Aggrigation  *********************
	// remove the residual inconsistencies in the focus volume
	if (DO_AGGRIGATION){
		cout << "Running Aggrigation"<< endl;

		//calculate the median CSTD for the whole volume
		vector<float> focus_curve_temp;
		vector<float> focus_curve_sum;
		focus_curve_sum = extractFocusCurve(AHO, 0, 0);	//extract the first curve

		for (int pixel_row = 0; pixel_row < num_rows; pixel_row++){
			for (int pixel_col = 0; pixel_col < num_cols; pixel_col++)
			{

				focus_curve_temp = extractFocusCurve(AHO, pixel_row, pixel_col);	//extract the next curve

				for (int image_no = 0; image_no < AHO.size(); image_no++){

					focus_curve_sum[image_no] = focus_curve_sum[image_no] + focus_curve_temp[image_no];
				}

				focus_curve_temp.clear();
			}
		}

		//Normalise
		for (int image_no = 0; image_no < AHO.size(); image_no++){
			focus_curve_sum[image_no] = focus_curve_sum[image_no] / (AHO[0].cols*AHO[0].rows);
		}
		//get the median CSTD
		float median_stdev, median_alpha;
		calcCSTD(focus_curve_sum, z_pos_vector, fm_vector_norm, z_step, decay_rate, median_stdev, median_alpha);

		Mat agg_filter = Mat::ones(agg_filter_size, agg_filter_size, CV_32F);
		Mat aggregation_weights = Mat::zeros(AHO[0].rows, AHO[0].cols, CV_32F);	//initialise a matrix to store the weights

		//for t iterations do the aggregation
		for (int t = 0; t < agg_iterations; t++){
			//calculate the weight matrix for each pixel x and y
			//for each pixel of each image in the volume
			parallel_for(0, num_rows, [&, num_cols](int pixel_row){
				for (int pixel_col = 0; pixel_col < num_cols; pixel_col++)
				{
					//extract the vector of focus measures from the volume
					vector<float> ag_fm_vec;
					vector<float> ag_fm_vec_norm;
					ag_fm_vec = extractFocusCurve(focus_measure_stack, pixel_row, pixel_col);

					//calculate the curve standard deviation (CSTD)
					calcCSTD(ag_fm_vec, z_pos_vector, ag_fm_vec_norm, z_step, decay_rate, curve_stdev, alpha);

					aggregation_weights.at<float>(pixel_row, pixel_col) = 1 / (1 + ((curve_stdev - median_stdev) / decay_rate)*((curve_stdev - median_stdev) / decay_rate));

					ag_fm_vec.clear();
					ag_fm_vec_norm.clear();

				}
			});

			//given a pixel location sum the pixels in the window * their weights
			//output = A.mul(B); element wise multiply each image by the weights

			for (int image_no = 0; image_no < AHO.size(); image_no++){
				//normaliseAndDisplay(AHO, image_no, "AHO_agg");
				AHO[image_no] = AHO[image_no].mul(aggregation_weights);

				cv::filter2D(AHO[image_no], AHO[image_no], CV_32F, agg_filter); //filter 2d the weighted AHO frames with a ones filter
				//normaliseAndDisplay(AHO, image_no, "AHO_agg");
			}

		} //aggrigation loop

		//save the outputs of the image aggrigation step
		string aggrigation_filename = "aggrigation_result.yml";
		FileStorage fs3(aggrigation_filename, FileStorage::WRITE);
		fs3 << "AHO" << AHO;
		fs3.release();
	}

	//else{ //if we haven't run the filtering read the data from disk	
	////TODO: this should be if we havent run aggrigation just use the pre aggrigation AHO
	//	string AHO_filename = "aggrigation_result.yml";
	//	FileStorage fs2(AHO_filename, FileStorage::READ);
	//	fs2["AHO"] >> AHO;
	//}

	//do the reconstruction
	reconstruction(AHO, input_image_stack, z_pos_vector, depthmap, colormap);
	


	//save the result to file
	FileStorage fs("scan_result.yml", FileStorage::WRITE);
	fs << "folderpath" << folderpath;

	fs << "decay_rate" << decay_rate;		//hyperparameters
	fs << "z_step" << z_step;
	fs << "max_derivative_order" << max_derivative_order;
	fs << "kernel_size" << kernel_size;
	fs << "agg_iterations" << agg_iterations;
	fs << "agg_filter_size" << agg_filter_size;
	fs << "rec_threshold" << rec_threshold;

	fs << "depthmap" << depthmap;
	fs << "colormap" << colormap;
		
	fs.release();

	//normalise the new alpha_mat and display
	Mat alpha_mat_norm;
	normalize(alpha_mat, alpha_mat_norm, 1.0, 0.0, NORM_MINMAX);
	imshow("alpha_mat", alpha_mat);

	Mat depthmap_norm;
	normalize(depthmap, depthmap_norm, 1.0, 0.0, NORM_MINMAX);

	//show the normalised depthmap and display a focus curve based on pixel selection
	namedWindow("depthmap_norm", 0); // Can be resized

	cv::setMouseCallback("depthmap_norm", onMouse, (void*)&pt);

	for (;;)
	{
		cv::imshow("depthmap_norm", depthmap_norm);

		char c = (char)waitKey(0);

		if (c == 27)
		{
			cout << "Exiting ...\n";
			break;
		}
	}


	//display the all in focus colormap
	imshow("colormap", colormap);
	waitKey(0);

	//display the 3D data
	viz_plot viz;
	viz.vizPlot3D(depthmap, colormap, pixel_width_mm, pixel_height_mm);

  }
