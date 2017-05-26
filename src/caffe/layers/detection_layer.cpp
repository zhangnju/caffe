#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include "caffe/layer.hpp"
#include "caffe/layers/detection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DetectionLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	//reshpe top blob
	vector<int> box_shape(4);
	box_shape[0] = num_object_; box_shape[1] = width_;
	box_shape[2] = height_; box_shape[3] = coords_;
	top[0]->Reshape(box_shape);

	vector<int> prob_shape(4);
	prob_shape[0] = num_object_; prob_shape[1] = width_;
	prob_shape[2] = height_; prob_shape[3] = num_class_+1;
	top[1]->Reshape(prob_shape);
}

template <typename Dtype>
void DetectionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Layer<Dtype>::LayerSetUp(bottom, top);
  DetectionLossParameter param = this->layer_param_.detection_loss_param();
  width_ = param.side();
  height_ = param.side();
  coords_ = param.coords(); //4
  softmax_ = param.softmax(); //
  batch_ = param.batch();;//check me 
  num_class_ = param.num_class();
  num_object_ = param.num_object();

  int input_count = bottom[0]->count(1);
  // outputs: classes, iou, coordinates
  int tmp_input_count = width_ * height_ * (num_class_ + (1 + coords_) * num_object_);
  CHECK_EQ(input_count, tmp_input_count);
}
/*
template <typename Dtype>
inline void softmax(Dtype *input, int n)
{
	Dtype sum = 0;
	Dtype largest = -FLT_MAX;
	for (int i = 0; i < n; ++i){
		if (input[i] > largest) largest = input[i];
	}
	for (int i = 0; i < n; ++i){
		sum += exp(input[i] - largest);
	}
	if (sum) sum = largest + log(sum);
	else sum = largest - 100;
	for (int i = 0; i < n; ++i){
		input[i] = exp(input[i] - sum);
	}
}*/
template <typename Dtype>
void DetectionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* input_data = bottom[0]->mutable_cpu_data();
  Dtype* box_data = top[0]->mutable_cpu_data();//check the size is right
  Dtype* prob_data = top[1]->mutable_cpu_data();
  if (softmax_){
	  for (int b = 0; b < batch_; ++b){
		  int index = b*width_*height_*((1 + coords_)*num_object_ + num_class_);
		  for (int i = 0; i < width_*height_; ++i) {
			  int offset = i*num_class_;
			  softmax_op(input_data + index + offset, num_class_,1);
		  }
	  }
  }

  for (int i = 0; i < width_*height_; ++i){
	  int row = i / width_;
	  int col = i % width_;
	  for (int n = 0; n < num_object_; ++n){
		  int index = i*num_object_ + n;
		  int p_index = width_*height_*num_class_ + i*num_object_ + n;
		  Dtype scale = input_data[p_index];
		  int box_index = width_*height_*(num_class_ + num_object_) + (i*num_object_ + n) * 4;
		  box_data[index] = (input_data[box_index + 0] + col) / width_; //check me ,here need to multiplied by image width
		  box_data[index+1] = (input_data[box_index + 1] + row) / width_;//check me ,here need to multiplied by image height
		  box_data[index+2] = pow(input_data[box_index + 2], (sqrt_ ? 2 : 1));
		  box_data[index+3] = pow(input_data[box_index + 3], (sqrt_ ? 2 : 1));
		  Dtype max_prob = 0;
		  for (int j = 0; j < num_class_; ++j){
			  int class_index = i*num_class_;
			  Dtype prob = scale*input_data[class_index + j];
			  prob_data[index*(num_class_+1) + j] = (prob > thresh_) ? prob : 0;
			  if (prob > max_prob) max_prob = prob;
		  }
		  prob_data[index*(num_class_ + 1) + num_class_] = max_prob>thresh_ ? max_prob : 0;
	  }
  }
}

INSTANTIATE_CLASS(DetectionLayer);
REGISTER_LAYER_CLASS(Detection);

}  // namespace caffe
