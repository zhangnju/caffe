#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include <iostream>
#include "caffe/layer.hpp"
#include "caffe/layers/region_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
void RegionLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	NOT_IMPLEMENTED;
}

template <typename Dtype>
inline void softmax_cpu(Dtype *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride)
{
	for (int b = 0; b < batch; ++b){
		for (int g = 0; g < groups; ++g){
			softmax_op(input + b*batch_offset + g*group_offset, n, stride);
		}
	}
}

template <typename Dtype>
vector<Dtype> get_region_box(Dtype* x, vector<Dtype> biases, int n, int index, int i, int j, int w, int h , int stride){
  vector<Dtype> b;
  b.push_back((i + x[index + 0*stride]) / w);
  b.push_back((j + x[index + 1*stride]) / h);
  b.push_back(exp(x[index + 2*stride]) * biases[2*n] / w);
  b.push_back(exp(x[index + 3*stride]) * biases[2*n+1] / h);
  return b;
}

template <typename Dtype>
inline Dtype sigmoid(Dtype x)
{
	return 1. / (1. + exp(-x));
}

template <typename Dtype>
void RegionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Layer<Dtype>::LayerSetUp(bottom, top);
  
  RegionLossParameter param = this->layer_param_.region_loss_param();
  
  height_ = param.side(); //13
  width_ =  param.side(); //13
  bias_match_ = param.bias_match(); //
  num_class_ = param.num_class(); //20
  coords_ = param.coords(); //4
  num_ = param.num(); //5
  softmax_ = param.softmax(); //
  batch_ = 1;//check me 
  jitter_ = param.jitter(); 
  rescore_ = param.rescore();
  
  for (int c = 0; c < param.biases_size(); ++c) {
	  biases_.push_back(param.biases(c));
  } //0.73 0.87;2.42 2.65;4.30 7.04;10.24 4.59;12.68 11.87;
  
  absolute_ = param.absolute();
  thresh_ = param.thresh(); //0.6
  random_ = param.random();  

  int input_count = bottom[0]->count(1); //h*w*n*(classes+coords+1) = 13*13*5*(20+4+1)
  // outputs: classes, iou, coordinates
  int tmp_input_count = width_ * height_ * num_ * (coords_ + num_class_ + 1); //13*13*5*(20+4+1)
  CHECK_EQ(input_count, tmp_input_count);
}


template <typename Dtype>
void RegionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	Dtype* input_data = bottom[0]->mutable_cpu_data();
	Dtype* box_data = top[0]->mutable_cpu_data();
	Dtype* prob_data = top[1]->mutable_cpu_data();
    
	for (int b = 0; b < batch_; b++)
	{
		for (int n = 0; n < num_; n++)
		{
			int index = entry_index(b, n*width_*height_, 0);
			for (int k = 0; k < 2 * width_*height_; k++)
			{
				input_data[index+k]=sigmoid(input_data[index+k]);
			}
			index = entry_index(b, n*width_*height_, 4);
			for (int k = 0; k < width_*height_; k++)
			{
				input_data[index+k] = sigmoid(input_data[index+k]);
			}
		}
	}

	if (softmax_)
	{
		int index = entry_index(0, 0, 5);
		softmax_cpu(input_data + index, num_class_, batch_*num_, height_*width_*(num_class_ + coords_ + 1), width_*height_, width_*height_,1);
	}
	
	for (int i = 0; i < width_*height_; ++i){
		int row = i / width_;
		int col = i % width_;
		for (int n = 0; n < num_; ++n){
			int index = n*width_*height_ + i;
			for (int j = 0; j < num_class_; ++j){
				prob_data[index + j] = 0;
			}
			int obj_index = entry_index(0, n*width_*height_ + i, 4);
			int box_index = entry_index(0, n*width_*height_ + i, 0);
			float scale = input_data[obj_index];
		 
			vector<Dtype > box= get_region_box(input_data, biases_, n, box_index, col, row, width_, height_, width_*height_);
			for (int k = 0; k < box.size(); k++)
				box_data[index + k] = box[k];
			int class_index = entry_index(0, n*width_*height_ + i, 5);
			
			float max_prob = 0;
			for (int j = 0; j < num_class_; ++j){
				int class_index = entry_index(0, n*width_*height_ + i, 5 + j);
				float prob = scale*input_data[class_index];
				prob_data[index+j] = (prob > thresh_) ? prob : 0;
				if (prob > max_prob) max_prob = prob;
				}
				prob_data[index+num_class_] = max_prob;
			}
		}
	 //add the function of correct_box_data here ?
}
INSTANTIATE_CLASS(RegionLayer);
REGISTER_LAYER_CLASS(Region);

}  // namespace caffe
