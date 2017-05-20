#ifndef CAFFE_REGION_LOSS_LAYER_HPP_
#define CAFFE_REGION_LOSS_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include <string>
#include "caffe/layers/loss_layer.hpp"
#include <map>

namespace caffe {
template <typename Dtype>
class RegionLossLayer : public LossLayer<Dtype> {
 public:
  explicit RegionLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RegionLoss"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  int height_;
  int width_;
  int bias_match_;
  int num_class_;
  int coords_;
  int num_;
  int softmax_;
  int batch_;
  float jitter_;
  int rescore_;
  
  float object_scale_;
  float class_scale_;
  float noobject_scale_;
  float coord_scale_;
  
  int absolute_;
  float thresh_;
  int random_;
  vector<Dtype> biases_;

  Blob<Dtype> diff_;
  
  inline int entry_index(int batch, int location, int entry)
  {
	  int n = location / (width_*height_);
	  int loc = location % (width_*height_);
	  return batch*(height_*width_*num_*(num_class_ + coords_ + 1)) + n*width_*height_*(coords_ + num_class_ + 1) + entry*width_*height_ + loc;
  }
};

}  // namespace caffe

#endif  // CAFFE_REGION_LOSS_LAYER_HPP_
