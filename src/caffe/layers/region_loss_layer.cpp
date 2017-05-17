#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

#include "caffe/layers/region_loss_layer.hpp"
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
  
  object_scale_ = param.object_scale(); //5.0
  noobject_scale_ = param.noobject_scale(); //1.0
  class_scale_ = param.class_scale(); //1.0
  coord_scale_ = param.coord_scale(); //1.0
  
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
void RegionLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const Dtype* label_data = bottom[1]->cpu_data(); //[label,x,y,w,h]
	Dtype* input_data = bottom[0]->mutable_cpu_data();
    Dtype* diff = diff_.mutable_cpu_data();
    caffe_set(diff_.count(), Dtype(0.0), diff);
    Dtype avg_anyobj(0.0), avg_obj(0.0), avg_iou(0.0), avg_cat(0.0), recall(0.0), loss(0.0);
    int count = 0;
    int class_count = 0;
    
	for (int b = 0; b < batch_; b++)
	{
		for (int n = 0; n < num_; n++)
		{
			int index = entry_index(b, n*width_*height_, 0);
			for (int k = 0; k < 2 * width_*height_; k++)
			{
				input_data[index]=sigmoid(input_data[index]);
			}
			index = entry_index(b, n*width_*height_, 4);
			for (int k = 0; k < width_*height_; k++)
			{
				input_data[index] = sigmoid(input_data[index]);
			}
		}
	}

	if (softmax_)
	{
		int index = entry_index(0, 0, 5);
		softmax_cpu(input_data + index, num_class_, batch_*num_, height_*width_*(num_class_ + coords_ + 1), width_*height_, width_*height_,1);
	}
	

	for (int b = 0; b < batch_; b++)
	{
		for (int j = 0; j < height_; ++j) {
			for (int i = 0; i < width_; ++i) {
				for (int n = 0; n < num_; ++n) {
					int box_index = entry_index(b, n*width_*height_ + j*width_ + i, 0);
					vector<Dtype> pred = get_region_box(input_data, biases_, n, box_index, i, j, width_, height_, width_*height_);
					Dtype best_iou = 0;
					for (int t = 0; t < 30; ++t){
						vector<Dtype> truth;
						truth.push_back(label_data[t * 5 + b * 30 * 5+1]);
						truth.push_back(label_data[t * 5 + b * 30 * 5+2]);
						truth.push_back(label_data[t * 5 + b * 30 * 5+3]);
						truth.push_back(label_data[t * 5 + b * 30 * 5+4]);
						if (truth[2]==0) break;
						Dtype iou = Calc_iou(pred, truth);
						if (iou > best_iou) {
							best_iou = iou;
						}
					}
					int obj_index = entry_index(b, n*width_*height_ + j*width_ + i, 4);
					avg_anyobj += input_data[obj_index];
					diff[obj_index] = noobject_scale_ * (0 - input_data[obj_index]);
					if (best_iou > thresh_) {
						diff[obj_index] = 0;
					}
				}
			}
		}

		for (int t = 0; t < 30; ++t){
			vector<Dtype> truth;
			int class_label = label_data[t * 5 + b * 30 * 5 + 0];
			truth.push_back(label_data[t * 5 + b * 30 * 5 + 1]);
			truth.push_back(label_data[t * 5 + b * 30 * 5 + 2]);
			truth.push_back(label_data[t * 5 + b * 30 * 5 + 3]);
			truth.push_back(label_data[t * 5 + b * 30 * 5 + 4]);
			if (truth[2]==0) break;
			float best_iou = 0;
			int best_n = 0;
			int i = truth[0] * width_; //match which i,j
			int j = truth[1] * height_;
			
			vector<Dtype> truth_shift = truth;
			truth_shift[0]=0;
			truth_shift[1]=0;

			for (int n = 0; n < num_; ++n){ //search 5 anchor in i,j
				int index = entry_index(b, n*width_*height_ + j*width_ + i, 0);
				vector<Dtype> pred = get_region_box(input_data, biases_, n, index, i, j, width_, height_,1); 
				if (bias_match_){
					pred[2] = biases_[2 * n] / width_;
					pred[3] = biases_[2 * n + 1] / height_;
				}
				pred[0] = 0;
				pred[1] = 0;
				float iou = Calc_iou(pred, truth_shift);
				if (iou > best_iou){
					best_iou = iou;
					best_n = n;
				}
			}
			int box_index = entry_index(b, best_n*width_*height_ + j*width_ + i, 0);
			Dtype iou = delta_region_box(truth, input_data, biases_, best_n, box_index, i, j, width_, height_, diff, coord_scale_ *  (2 - truth[2]*truth[3]), width_*height_);
			if (iou > .5) recall += 1;
			avg_iou += iou;

			int obj_index = entry_index(b, best_n*width_*height_ + j*width_ + i, 4);
			avg_obj += input_data[obj_index];
			diff[obj_index] = object_scale_ * (1 - input_data[obj_index]);
			if (rescore_) {
				diff[obj_index] = object_scale_ * (iou - input_data[obj_index]);
			}

			class_label = label_data[t * 5 + b*30*5 + 4];
			int class_index = entry_index(b, best_n*width_*height_ + j*width_ + i, 5);
			delta_region_class(input_data, diff, class_index, class_label, num_class_, class_scale_, width_*height_, &avg_cat);

			++count;
			++class_count;
		}
	}

	for (int i = 0; i < diff_.count(); ++i)
	{
		loss += diff[i] * diff[i];
	}
	top[0]->mutable_cpu_data()[0] = loss;
    	
    iter ++;
    LOG(INFO) << "iter: " << iter <<" loss: " << loss;
    LOG(INFO) << "avg_noobj: "<< avg_anyobj/(width_*height_*num_*bottom[0]->num()) << " avg_obj: " << avg_obj/count <<" avg_iou: " << avg_iou/count << " avg_cat: " << avg_cat/class_count << " recall: " << recall/count << " class_count: "<< class_count;
}

template <typename Dtype>
void RegionLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //LOG(INFO) <<" propagate_down: "<< propagate_down[1] << " " << propagate_down[0];
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype sign(1.);
    const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->num();
    //const Dtype alpha(1.0);
    //LOG(INFO) << "alpha:" << alpha;
    
    caffe_cpu_axpby(
        bottom[0]->count(),
        alpha,
        diff_.cpu_data(),
        Dtype(0),
        bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
//STUB_GPU(DetectionLossLayer);
#endif

INSTANTIATE_CLASS(RegionLossLayer);
REGISTER_LAYER_CLASS(RegionLoss);

}  // namespace caffe
