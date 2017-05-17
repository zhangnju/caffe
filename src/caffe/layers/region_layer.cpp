#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include <iostream>

#include "caffe/layers/region_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

int iter = 0;

namespace caffe {
template <typename Dtype>
Dtype Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2) {
  Dtype left = std::max(x1 - w1 / 2, x2 - w2 / 2);
  Dtype right = std::min(x1 + w1 / 2, x2 + w2 / 2);
  return right - left;
}

template <typename Dtype>
Dtype Calc_iou(const vector<Dtype>& box, const vector<Dtype>& truth) {
  Dtype w = Overlap(box[0], box[2], truth[0], truth[2]);
  Dtype h = Overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0) return 0;
  Dtype inter_area = w * h;
  Dtype union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return inter_area / union_area;
}

template <typename Dtype>
inline Dtype sigmoid(Dtype x)
{
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
Dtype softmax(Dtype* input, int classes, int stride)
{
  Dtype sum = 0;
  Dtype large = input[0];
  for (int i = 0; i < classes; ++i){
    if (input[i*stride] > large)
      large = input[i*stride];
  }
  for (int i = 0; i < classes; ++i){
    Dtype e = exp(input[i*stride] - large);
    sum += e;
    input[i*stride] = e;
  }
  for (int i = 0; i < classes; ++i){
    input[i*stride] = input[i*stride] / sum;
  }
  return 0;
}
template <typename Dtype>
inline void softmax_cpu(Dtype *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride)
{
	for (int b = 0; b < batch; ++b){
		for (int g = 0; g < groups; ++g){
			softmax(input + b*batch_offset + g*group_offset, n, stride);
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
Dtype delta_region_box(vector<Dtype> truth, Dtype* x, vector<Dtype> biases, int n, int index, int i, int j, int w, int h, Dtype* delta, float scale, int stride){
  vector<Dtype> pred;
  pred = get_region_box(x, biases, n, index, i, j, w, h,stride);
        
  float iou = Calc_iou(pred, truth);
  
  float tx = truth[0] * w - i; //0.5
  float ty = truth[1] * h - j; //0.5
  float tw = log(truth[2] * w / biases[2*n]); //truth[2]=biases/w tw = 0
  float th = log(truth[3] * h / biases[2*n + 1]); //th = 0
	
  delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
  delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
  delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
  delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
  return iou;
}

template <typename Dtype>
void delta_region_class(Dtype* input_data, Dtype* diff, int index, int class_label, int classes, float scale, int stride, Dtype* avg_cat)
{
    for (int n = 0; n < classes; ++n){
      diff[index + stride*n] = scale * (((n == class_label)?1 : 0) - input_data[index + stride*n]);
      if (n == class_label){
        *avg_cat += input_data[index + stride*n];
      }
    }
}

template <typename Dtype>
void RegionLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  
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
  

  
  absolute_ = param.absolute();
  thresh_ = param.thresh(); //0.6
  random_ = param.random();  

  for (int c = 0; c < param.biases_size(); ++c) {
     biases_.push_back(param.biases(c)); 
  } //0.73 0.87;2.42 2.65;4.30 7.04;10.24 4.59;12.68 11.87;

  diff_.ReshapeLike(*bottom[0]);
  int input_count = bottom[0]->count(1); //h*w*n*(classes+coords+1) = 13*13*5*(20+4+1)
  int label_count = bottom[1]->count(1); //30*5
  // outputs: classes, iou, coordinates
  int tmp_input_count = width_ * height_ * num_ * (coords_ + num_class_ + 1); //13*13*5*(20+4+1)
  int tmp_label_count = 30 * num_;
  CHECK_EQ(input_count, tmp_input_count);
  CHECK_EQ(label_count, tmp_label_count);
}


template <typename Dtype>
void RegionLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void RegionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const Dtype* label_data = bottom[1]->cpu_data(); //[label,x,y,w,h]
	Dtype* input_data = bottom[0]->mutable_cpu_data();
    
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
	

}
INSTANTIATE_CLASS(RegionLayer);
REGISTER_LAYER_CLASS(Region);

}  // namespace caffe
