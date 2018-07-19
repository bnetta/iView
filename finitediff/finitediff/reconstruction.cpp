#include "reconstruction.h"
#include "vector_tools.h"

using namespace std;
using namespace cv;


void reconstruction::reconCentroid(vector<Mat> AHO, vector<Mat> input_image_stack, vector<float> z_pos_vector, Mat& depthmap, Mat& colormap) {
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
				if (alpha > rec_alpha_threshold) {

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
					depthmap.at<float>(pixel_row, pixel_col) = NAN;
					colormap.at<unsigned char>(pixel_row, pixel_col) = NAN;
				}


				rec_fm_vec.clear();
				rec_fm_vec_norm.clear();
				top_fm_vector.clear();
				top_fm_z_pos_vector.clear();

			}
		}); //reconstruction loop
	}