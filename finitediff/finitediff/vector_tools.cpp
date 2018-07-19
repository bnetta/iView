#include "vector_tools.h"

using namespace std;
using namespace cv;

//get the maximum value in a vector TODO: find a standard way to do this
float vectorMax(vector<float>vector){

	float val_max = std::numeric_limits<float>::min();;
	float val;

	for (int i = 0; i < vector.size(); i++){
		val = vector[i];
		if (val > val_max){
			val_max = val;
		}
	}
	return val_max;
}

//calculate the curve standard deiation (CSTD) of a focus curve
void calcCSTD(vector<float> fm_vector, vector<float> z_pos_vector, vector<float>& fm_vector_norm, float z_step, float decay_rate, float& curve_stdev, float& alpha){

	float fm_val_max = vectorMax(fm_vector);
	vector<float> top_fm_vector;
	vector<float> top_fm_z_pos_vector;

	//create a vector with only entries >50% of max value and normalise
	for (int image_no = 0; image_no < fm_vector.size(); image_no++){

		//normalise whole vector
		fm_vector_norm.push_back(fm_vector[image_no] / fm_val_max);

		//add values over half the maximum to a new vector
		if (fm_vector[image_no]>(fm_val_max / 2)){
			top_fm_vector.push_back(fm_vector[image_no] / fm_val_max);
			top_fm_z_pos_vector.push_back(z_pos_vector[image_no]);
		}
	}

	float fm_norm_sum = std::accumulate(fm_vector_norm.begin(), fm_vector_norm.end(), 0.0);

	float sum_top = std::accumulate(top_fm_vector.begin(), top_fm_vector.end(), 0.0);

	float mean_top = sum_top / top_fm_vector.size();

	//calculate the centroid of rotation about the fm axis
	float z_centroid = 0;
	float inertia_sum = 0;
	for (int i_z = 0; i_z < top_fm_vector.size(); i_z++){
		inertia_sum += (top_fm_vector[i_z] * z_step) * top_fm_z_pos_vector[i_z];
	}

	z_centroid = inertia_sum / (sum_top * z_step);

	float sq_sum = 0;
	for (int i_z = 0; i_z < top_fm_vector.size(); i_z++){
		sq_sum = sq_sum + (top_fm_z_pos_vector[i_z] - z_centroid) * (top_fm_z_pos_vector[i_z] - z_centroid) * top_fm_vector[i_z];
	}

	curve_stdev = (1 / sqrt(sum_top*z_step))*sqrt(sq_sum);

	if (!isnormal(curve_stdev)){
		curve_stdev = 0;
		alpha = 0;
	}
	else{
		alpha = 1 / (1 + (curve_stdev / (decay_rate))*(curve_stdev / (decay_rate))); //TODO: decide if this is valid to include z_step here
	}
	top_fm_vector.clear();
	top_fm_z_pos_vector.clear();
}

//extract a given focus curve from the volume
std::vector<float> extractFocusCurve(std::vector<Mat>focus_volume, int row_index, int col_index){

	std::vector<float> fm_vector;

	//extract the vector of focus measures from the volume
	for (int image_no = 0; image_no < focus_volume.size(); image_no++){
		float fm_val = focus_volume[image_no].at<float>(row_index, col_index);
		fm_vector.push_back(fm_val);
	}

	return fm_vector;
}