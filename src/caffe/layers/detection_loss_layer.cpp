#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layers/detection_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

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
Dtype Calc_rmse(const vector<Dtype>& box, const vector<Dtype>& truth) {
  return sqrt(pow(box[0]-truth[0], 2) +
              pow(box[1]-truth[1], 2) +
              pow(box[2]-truth[2], 2) +
              pow(box[3]-truth[3], 2));
}

template <typename Dtype>
void DetectionLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  DetectionLossParameter param = this->layer_param_.detection_loss_param();
  width_ = param.side();
  height_ = param.side();
  num_class_ = param.num_class();
  num_object_ = param.num_object();
  sqrt_ = param.sqrt();
  coords_ = param.coords(); //4
  softmax_ = param.softmax(); //
  batch_ = param.batch();//check me 
  rescore_ = param.rescore();
  object_scale_ = param.object_scale();
  noobject_scale_ = param.noobject_scale();
  class_scale_ = param.class_scale();
  coord_scale_ = param.coord_scale();
  
  int input_count = bottom[0]->count(1);
  int label_count = bottom[1]->count(1);
  // outputs: classes, iou, coordinates
  int tmp_input_count = width_ * height_ * (num_class_ + (1 + 4) * num_object_);
  // label: isobj, class_label, coordinates
  //int tmp_label_count = side_ * side_ * (1 + 1 + 1 + 4);？
  int tmp_label_count = width_ * height_ * (1 + num_class_ + 4);
  CHECK_EQ(input_count, tmp_input_count);
  CHECK_EQ(label_count, tmp_label_count);
}

template <typename Dtype>
void DetectionLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void DetectionLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    Dtype* input_data = bottom[0]->mutable_cpu_data();
    const Dtype* label_data = bottom[1]->cpu_data();
    Dtype* diff = diff_.mutable_cpu_data();
    Dtype loss(0.0);
    Dtype avg_iou(0.0), avg_cat(0.0), avg_allcat(0.0), avg_obj(0.0), avg_anyobj(0.0);
    Dtype obj_count(0);
    int locations = pow(width_, 2);
    caffe_set(diff_.count(), Dtype(0.), diff);

    if (softmax_){
	    for (int b = 0; b < batch_; ++b){
		   int index = b*width_*height_*((1 + coords_)*num_object_ + num_class_);
		   for (int i = 0; i < width_*height_; ++i) {
			   int offset = i*num_class_;
			   softmax_op(input_data + index + offset, num_class_, 1);
		   }
	    }
    }

    for (int b = 0; b < batch_; ++b) {
	    int index = b * width_*height_*((1 + coords_)*num_object_ + num_class_);
        for (int i = 0; i < locations; ++i) {
		   int true_index = (b * locations + i)*(1 + num_class_ + coords_);
           for (int j = 0; j < num_object_; ++j) {
		     int p_index = index + num_class_ * locations + i* num_object_ + j;
             loss += noobject_scale_ * pow(input_data[p_index], 2);
		     diff[p_index] = noobject_scale_ * (0 - input_data[p_index]);
             avg_anyobj += input_data[p_index];
           }

           int isobj = label_data[true_index];
           if (!isobj) {
             continue;
           }

           obj_count += 1;
		   int class_index = index + i*num_class_;
           for (int j = 0; j < num_class_; ++j) {
			  diff[class_index] = class_scale_ * (label_data[true_index + 1 + j] - input_data[class_index+j]);
			  loss += class_scale_ * pow(label_data[true_index + 1 + j] - input_data[class_index+j], 2);
              avg_allcat += input_data[class_index+j];
		      if (label_data[true_index + 1 + j])
                  avg_cat += input_data[class_index+j]; 
		   }

          const Dtype* true_box_pt = label_data + true_index + 1 + num_class_;
		  vector<Dtype> true_box; 
		  true_box.push_back((*true_box_pt)/width_);
		  true_box.push_back((*(true_box_pt + 1))/width_);
		  true_box.push_back(*(true_box_pt + 2));
		  true_box.push_back(*(true_box_pt + 3));
		  
      
          Dtype best_iou = 0.;
          Dtype best_rmse = 20.;
          int best_index = 0;

          for (int j = 0; j < num_object_; ++j) {
             vector<Dtype> box;
			 const Dtype* box_pt = input_data + index + (num_class_ + num_object_)*locations + (i*num_object_+j)*coords_;
             box.push_back(*box_pt/width_);
             box.push_back((*(box_pt + 1))/width_);
             box.push_back(*(box_pt + 2));
             box.push_back(*(box_pt + 3));
       
             if (sqrt_) {
                 box[2] = pow(box[2], 2);
                 box[3] = pow(box[3], 2);
             }
           
			 Dtype iou = Calc_iou(box, true_box);
             Dtype rmse = Calc_rmse(box, true_box);
             if (best_iou > 0 || iou > 0) {
               if (iou > best_iou) {
                  best_iou = iou;
                  best_index = j;
               }
             } else {
                 if (rmse < best_rmse) {
                  best_rmse = rmse;
                  best_index = j;
                 }
             }
           }

		   int box_index = index + locations*(num_class_ + num_object_) + (i*num_object_ + best_index) * coords_;
		   int tbox_index = true_index + 1 + num_class_;
		   vector<Dtype> out;
		   out.push_back((*(input_data+box_index)) / width_);
		   out.push_back((*(input_data + box_index + 1)) / width_);
		   out.push_back(*(input_data + box_index + 2));
		   out.push_back(*(input_data + box_index + 3));
		   if (sqrt_) {
			  out[2] = out[2]*out[2];
			  out[3] = out[3]*out[3];
		   }
		   float iou = Calc_iou(out, true_box);

           int p_index = index + num_class_ * locations + i*num_object_ + best_index;
           loss -= noobject_scale_ * pow(input_data[p_index], 2);
           loss += object_scale_ * pow(1.0-input_data[p_index], 2);
           avg_obj += input_data[p_index];
	       diff[p_index] = object_scale_ * (1.0 - input_data[p_index]);
           // rescore
	       if (rescore_)
               diff[p_index] = object_scale_ * (best_iou - input_data[p_index]);
     
	 
           for (int k = 0; k < 4; ++k) {
             diff[box_index + k ] = coord_scale_ * (label_data[tbox_index+k] - input_data[box_index+k]);
           }
	       if (sqrt_) {
			   diff[box_index + 2] = coord_scale_ * (sqrt(label_data[tbox_index + 2]) - input_data[box_index + 2]);
			   diff[box_index + 3] = coord_scale_ * (sqrt(label_data[tbox_index + 3]) - input_data[box_index + 3]);
	       }
		   loss += pow(1 - iou, 2);
		   avg_iou += iou;
        }
      }
    
	  float sum = 0;
	  for (int i = 0; i < diff_.count(); ++i)
	  {
		  sum += diff[i] * diff[i];
	  }
	  loss = sum;
      top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void DetectionLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype sign(1.);
    const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->num();
    caffe_cpu_axpby(
        bottom[0]->count(),
        alpha,
        diff_.cpu_data(),
        Dtype(0),
        bottom[0]->mutable_cpu_diff());
  }
}

INSTANTIATE_CLASS(DetectionLossLayer);
REGISTER_LAYER_CLASS(DetectionLoss);

}  // namespace caffe
