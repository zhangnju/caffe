#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/blob.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::string;
using caffe::vector;
using std::map;
using std::pair;

DEFINE_string(input, "",
	"Input image for run dection");
DEFINE_int32(type, 1,
	"The type of yolo:V1 is 1, and V2 is 2");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_double(nms, 0.40,
    "The thresh of nms.");
#if 0
template <typename T>
bool SortScorePairDescend(const pair<float, T>& pair1,
                          const pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

// Explicit initialization.
template bool SortScorePairDescend(const pair<float, int>& pair1,
                                   const pair<float, int>& pair2);
template bool SortScorePairDescend(const pair<float, pair<int, int> >& pair1,
                                   const pair<float, pair<int, int> >& pair2);

void CumSum(const vector<pair<float, int> >& pairs, vector<int>* cumsum) {
  // Sort the pairs based on first item of the pair.
  vector<pair<float, int> > sort_pairs = pairs;
  std::stable_sort(sort_pairs.begin(), sort_pairs.end(),
                   SortScorePairDescend<int>);

  cumsum->clear();
  for (int i = 0; i < sort_pairs.size(); ++i) {
    if (i == 0) {
      cumsum->push_back(sort_pairs[i].second);
    } else {
      cumsum->push_back(cumsum->back() + sort_pairs[i].second);
    }
  }
}

void ComputeAP(const vector<pair<float, int> >& tp, const int num_pos,
               const vector<pair<float, int> >& fp, const string ap_version,
               vector<float>* prec, vector<float>* rec, float* ap) {
  const float eps = 1e-6;
  CHECK_EQ(tp.size(), fp.size()) << "tp must have same size as fp.";
  const int num = tp.size();
  // Make sure that tp and fp have complement value.
  for (int i = 0; i < num; ++i) {
    CHECK_LE(fabs(tp[i].first - fp[i].first), eps);
    CHECK_GE(tp[i].second, 0);
    CHECK_GE(fp[i].second, 0);
  }
  prec->clear();
  rec->clear();
  *ap = 0;
  if (tp.size() == 0 || num_pos == 0) {
    return;
  }

  // Compute cumsum of tp.
  vector<int> tp_cumsum;
  CumSum(tp, &tp_cumsum);
  CHECK_EQ(tp_cumsum.size(), num);

  // Compute cumsum of fp.
  vector<int> fp_cumsum;
  CumSum(fp, &fp_cumsum);
  CHECK_EQ(fp_cumsum.size(), num);

  // Compute precision.
  for (int i = 0; i < num; ++i) {
    prec->push_back(static_cast<float>(tp_cumsum[i]) /
                    (tp_cumsum[i] + fp_cumsum[i]));
  }

  // Compute recall.
  for (int i = 0; i < num; ++i) {
    CHECK_LE(tp_cumsum[i], num_pos);
    rec->push_back(static_cast<float>(tp_cumsum[i]) / num_pos);
  }

  // for (int i = 0; i < num; ++i) {
  //   std::cout << (*prec)[i] << std::endl;
  //   std::cout << (*rec)[i] << std::endl;
  // }

  if (ap_version == "11point") {
    // VOC2007 style for computing AP.
    vector<float> max_precs(11, 0.);
    int start_idx = num - 1;
    for (int j = 10; j >= 0; --j) {
      for (int i = start_idx; i >= 0 ; --i) {
        if ((*rec)[i] < j / 10.) {
          start_idx = i;
          if (j > 0) {
            max_precs[j-1] = max_precs[j];
          }
          break;
        } else {
          if (max_precs[j] < (*prec)[i]) {
            max_precs[j] = (*prec)[i];
          }
        }
      }
    }
    for (int j = 10; j >= 0; --j) {
      *ap += max_precs[j] / 11;
    }
  } else if (ap_version == "MaxIntegral") {
    // VOC2012 or ILSVRC style for computing AP.
    float cur_rec = rec->back();
    float cur_prec = prec->back();
    for (int i = num - 2; i >= 0; --i) {
      cur_prec = std::max<float>((*prec)[i], cur_prec);
      if (fabs(cur_rec - (*rec)[i]) > eps) {
        *ap += cur_prec * fabs(cur_rec - (*rec)[i]);
      }
      cur_rec = (*rec)[i];
    }
    *ap += cur_rec * cur_prec;
  } else if (ap_version == "Integral") {
    // Natural integral.
    float prev_rec = 0.;
    for (int i = 0; i < num; ++i) {
      if (fabs((*rec)[i] - prev_rec) > eps) {
        *ap += (*prec)[i] * fabs((*rec)[i] - prev_rec);
      }
      prev_rec = (*rec)[i];
    }
  } else {
    LOG(FATAL) << "Unknown ap_version: " << ap_version;
  }
}
#endif
static int num_class = 0;
void preprocess_image(Net<float>& net,std::string& input, int width, int height)
{
	cv::Mat resized, resized_float;
	cv::Size size(width, height);
	cv::Mat orig_image = cv::imread(input, CV_LOAD_IMAGE_COLOR);
	if (width != orig_image.cols || height != orig_image.rows)
	{
		cv::resize(orig_image, resized, size);
	}
	else
	{
		resized = orig_image;
	}
	resized.convertTo(resized_float, CV_32FC3);

	Blob<float>* input_layer = net.input_blobs()[0];
	int num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	cv::Size input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net.Reshape();

	std::vector<cv::Mat> input_channels;
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += input_layer->width() * input_layer->height();
	}

	cv::split(resized_float, input_channels);
	for (int i = 0; i < input_layer->channels(); ++i) 
	   cv::normalize(input_channels[i], input_channels[i], 1.0, 0.0, cv::NORM_MINMAX);

	CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
		== net.input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";

}
typedef struct{
	int index_;
	int class_;
	float *probs_;
} sortable_bbox;

int nms_comparator(const void *pa, const void *pb)
{
	sortable_bbox a = *(sortable_bbox *)pa;
	sortable_bbox b = *(sortable_bbox *)pb;
	float diff = a.probs_[a.index_*(num_class+1)+a.class_] - b.probs_[b.index_*(num_class+1)+b.class_];
	if (diff < 0) return 1;
	else if (diff > 0) return -1;
	return 0;
}

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

std::string labels[20] = { "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
                          "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor" };

void draw_detections(std::string input, int num, const float *boxes, const float *probs, int classes)
{
	std::string prediction = "prediction.jpg";
	cv::Mat orig_image = cv::imread(input, CV_LOAD_IMAGE_COLOR);

	for (int i = 0; i < num; ++i){
		int class_id = -1;
		if (probs[i*(classes + 1) + classes]!=0){
			for (int j = 0; j < classes; j++)
				if (probs[i*(classes + 1) + classes] == probs[i*(classes + 1) + j])
					class_id = j;
			if (class_id == -1)
				continue;
			const float* b = &boxes[i*4];
			int left = (*b - *(b+2) / 2.)*orig_image.cols;
			int right = (*b + *(b+2) / 2.)*orig_image.cols;
			int top = (*(b+1) - *(b+3) / 2.)*orig_image.rows;
			int bot = (*(b+1) + *(b+3) / 2.)*orig_image.rows;

			if (left < 0) left = 0;
			if (right > orig_image.cols - 1) right = orig_image.cols - 1;
			if (top < 0) top = 0;
			if (bot > orig_image.rows - 1) bot = orig_image.rows - 1;

			cv::rectangle(orig_image, cv::Point(left, top), cv::Point(right, bot), cv::Scalar(0, 0, 0));
			cv::putText(orig_image, labels[class_id], cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(0, 0, 255),2.0);
		}
	}
	cv::imwrite(prediction, orig_image);
}
// Test: score a model.
int yolo_detection() {

    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
    CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

    #ifdef CPU_ONLY
       Caffe::set_mode(Caffe::CPU);
    #else
       Caffe::set_mode(Caffe::GPU);
    #endif

    int side ,resize_width,resize_height,num_object;
    if (FLAGS_type == 1)
    {
	   side = 7;
	   resize_width = 448;
	   resize_height = 448;
	   num_object = 2;
    }
    else if (FLAGS_type == 2)
    {
	   side = 13;
	   resize_width = 416;
	   resize_height = 416;
	   num_object = 5;
    }
    else
    {
	   LOG(ERROR) << "Wrong Yolo Version ";
    }
    num_class = 20;

    
    // Instantiate the caffe net.
    Net<float> caffe_net(FLAGS_model, caffe::TEST);
    caffe_net.CopyTrainedLayersFrom(FLAGS_weights);


    //preprocess image
	preprocess_image(caffe_net, FLAGS_input, resize_width, resize_height);
    
	const vector<Blob<float>*>& result = caffe_net.Forward();
    
    float* box_data = result[0]->mutable_cpu_data();
    int box_size = result[0]->count();
	float* prob_data = result[1]->mutable_cpu_data();
	int prob_size = result[1]->count();

    if (FLAGS_type == 2)
    {
	  //add the post-processing for yolo v2
	  int new_w = 0;
	  int new_h = 0;
	  cv::Mat orig_image = cv::imread(FLAGS_input, CV_LOAD_IMAGE_COLOR);
	  if (((float)resize_width / orig_image.cols) < ((float)resize_height / orig_image.rows)) {
		  new_w = resize_width;
		  new_h = (orig_image.rows * resize_width) / orig_image.cols;
	  }
	  else {
		  new_h = resize_height;
		  new_w = (orig_image.cols * resize_height) / orig_image.rows;
	  }
	  
	  for (int i = 0; i < box_size; i+=4){
		  box_data[i] = (box_data[i] - (resize_width - new_w) / 2. / resize_width) / ((float)new_w / resize_width);
		  box_data[i + 1] = (box_data[i + 1] - (resize_height - new_h) / 2. / resize_height) / ((float)new_h / resize_height);
		  box_data[i + 2] *= (float)resize_width / new_w;
		  box_data[i + 3] *= (float)resize_height / new_h;
	  }

	  if (FLAGS_nms)
	  {
		  //check me, shall we need to improve it?
		  sortable_bbox *s = (sortable_bbox *)new sortable_bbox[side*side*num_object];

		  for (int i = 0; i < side*side*num_object; ++i){
			  s[i].index_ = i;
			  s[i].class_ = num_class;
			  s[i].probs_ = prob_data;
		  }

		  qsort(s, side*side*num_object, sizeof(sortable_bbox), nms_comparator);
		  for (int i = 0; i < side*side*num_object; ++i){
			  if (prob_data[s[i].index_*(num_class + 1) + num_class] == 0) continue;
			  std::vector<float> a;
			  a.push_back(box_data[s[i].index_ * 4 + 0]);
			  a.push_back(box_data[s[i].index_ * 4 + 1]);
			  a.push_back(box_data[s[i].index_ * 4 + 2]);
			  a.push_back(box_data[s[i].index_ * 4 + 3]);
			  for (int j = i + 1; j < side*side*num_object; ++j){
				  std::vector<float> b;
				  b.push_back(box_data[s[j].index_ * 4 + 0]);
				  b.push_back(box_data[s[j].index_ * 4 + 1]);
				  b.push_back(box_data[s[j].index_ * 4 + 2]);
				  b.push_back(box_data[s[j].index_ * 4 + 3]);
				  if (Calc_iou(a, b) > FLAGS_nms){
					  for (int k = 0; k < num_class + 1; ++k){
						  prob_data[s[j].index_*(num_class + 1) + k] = 0;
					  }
				  }
			  }
		  }
		  delete[]s;
	  }
    }
	else
	{
		if (FLAGS_nms)
		{
			sortable_bbox *s = (sortable_bbox *)new sortable_bbox[side*side*num_object];

			for (int i = 0; i < side*side*num_object; ++i){
				s[i].index_ = i;
				s[i].class_ = 0;
				s[i].probs_ = prob_data;
			}

			for (int k = 0; k < num_class; ++k){
				for (int i = 0; i < side*side*num_object; ++i){
					s[i].class_ = k;
				}
				qsort(s, side*side*num_object, sizeof(sortable_bbox), nms_comparator);
				for (int i = 0; i < side*side*num_object; ++i){
					if (prob_data[s[i].index_*(num_class + 1) + k] == 0) continue;
					std::vector<float> a;
					a.push_back(box_data[s[i].index_ * 4 + 0]);
					a.push_back(box_data[s[i].index_ * 4 + 1]);
					a.push_back(box_data[s[i].index_ * 4 + 2]);
					a.push_back(box_data[s[i].index_ * 4 + 3]);
					for (int j = i + 1; j < side*side*num_object; ++j){
						std::vector<float> b;
						b.push_back(box_data[s[j].index_ * 4 + 0]);
						b.push_back(box_data[s[j].index_ * 4 + 1]);
						b.push_back(box_data[s[j].index_ * 4 + 2]);
						b.push_back(box_data[s[j].index_ * 4 + 3]);
						if (Calc_iou(a, b) > FLAGS_nms){
							prob_data[s[j].index_*(num_class + 1) + k] = 0;
						}
					}
				}
			}
			delete[]s;
		}
	}
 
    draw_detections(FLAGS_input, side * side * num_object, result[0]->cpu_data(), result[1]->cpu_data(), num_class);
    std::cout << "OK" << std::endl;

     return 0;
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  // Usage message.
  gflags::SetUsageMessage("Test a object detection model\n"
        "Usage:\n"
        "    yolo_detection [FLAGS] \n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  
  return yolo_detection();
}
