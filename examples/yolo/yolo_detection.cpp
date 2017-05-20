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
DEFINE_int32(version, 1,
	"The version of yolo:V1 is 1, and V2 is 2");
DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 1,
    "The number of iterations to run.");

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
  if (FLAGS_gpu == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  } else if (FLAGS_gpu.size()) {
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus->size(), 0);
  }
}

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
int max_index(float *a, int n)
{
	if (n <= 0) return -1;
	int i, max_i = 0;
	float max = a[0];
	for (i = 1; i < n; ++i){
		if (a[i] > max){
			max = a[i];
			max_i = i;
		}
	}
	return max_i;
}

void resize_image(std::string& input,std::string& output,int width,int height)
{
	cv::Mat resize_image;
	cv::Size size(width, height);
	cv::Mat orig_image = cv::imread(input, CV_LOAD_IMAGE_COLOR);
	if (width!=orig_image.cols || height!=orig_image.rows)
	{
		cv::resize(orig_image, resize_image, size);
		cv::imwrite(output, resize_image);
	}
	else
	{
		output = input;
	}
		
}

// Test: score a model.
int test_detection() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  int side ,resize_width,resize_height;
  if (FLAGS_version == 1)
  {
	  side = 7;
	  resize_width = 448;
	  resize_height = 448;
  }
  else if (FLAGS_version == 2)
  {
	  side = 13;
	  resize_width = 416;
	  resize_height = 416;
  }
  else
  {
	  LOG(ERROR) << "Wrong Yolo Version ";
  }
  int num_object = 2;
  int num_class = 20;

  caffe::CPUTimer timer;
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);

#if 0
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  map<int, map<int, vector<pair<float, int> > > > true_poss, false_poss;
  map<int, map<int, int> > num_gts;
  vector<int> test_score_output_id;
  float loss = 0;
  int out_size(0);
  map<int, int> skip_idx;
  double total_time(0.0);
  timer.Start();
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result = caffe_net.Forward(&iter_loss);
    out_size = result.size();
    LOG(INFO) << "iter_loss: " << iter_loss;
    loss += iter_loss;
    for (int j = 0; j < result.size(); ++j) {
      if (result[j]->count() == 1) {
        skip_idx[j] = 0;
        continue;
      }
      const float* result_data = result[j]->cpu_data();
      for (int k = 0; k < result[j]->num(); ++k) {
        int res_index = k * result[j]->count(1);
        for (int c = 0; c < num_class; ++c) {
          if (num_gts[j].find(c) == num_gts[j].end()) {
            num_gts[j][c] = static_cast<int>(result_data[res_index + c]);
          } else {
            num_gts[j][c] += static_cast<int>(result_data[res_index + c]);
          }
        }
        int all_obj_num = side * side * num_object;
        int obj_index = res_index + num_class;
        for (int b = 0; b < all_obj_num; ++b) {
          int label = static_cast<int>(result_data[obj_index + b * 4 + 0]);
          float score = result_data[obj_index + b * 4 + 1];
          int tp = static_cast<int>(result_data[obj_index + b * 4 + 2]);
          int fp = static_cast<int>(result_data[obj_index + b * 4 + 3]);
          // LOG(INFO) << "tp: " << tp << " fp: " << fp;
          true_poss[j][label].push_back(std::make_pair(score, tp));
          false_poss[j][label].push_back(std::make_pair(score, fp));
        }
      }
    }
    LOG(INFO) << "Running Iteration " << i;
  }
  total_time += timer.MicroSeconds();
  LOG(INFO) << "Total time: " << total_time / 1000 << " ms.";
  for (int i = 0; i < out_size; ++i) {
    if (skip_idx.find(i) != skip_idx.end()) {
      continue;
    }
    map<int, vector<pair<float, int> > > true_pos = true_poss[i];
    map<int, vector<pair<float, int> > > false_pos = false_poss[i];
    map<int, int> num_gt = num_gts[i];
    map<int, float> APs;
    float mAP = 0.;
    for (int j = 0; j < num_class; ++j) {
      if (!num_gt[j]) {
        LOG(WARNING) << "Ground trurh label number is 0: " << j;
        continue;
      } 
      if (true_pos.find(j) == true_pos.end()) {
        LOG(WARNING) << "Missing true_pos for label: " << j;
        continue;
      }
      if (false_pos.find(j) == false_pos.end()) {
        LOG(WARNING) << "Missing false_pos for label: " << j;
        continue;
      }
      string ap_version = "11point";
      vector<float> prec, rec;
      ComputeAP(true_pos[j], num_gt[j], false_pos[j], ap_version, &prec, &rec, &(APs[j]));
      mAP += APs[j];
    }
    mAP /= num_class;
    const string& output_name = caffe_net.blob_names()[caffe_net.output_blob_indices()[i]];
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = " << mAP;
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  LOG(INFO) << "Model: " << FLAGS_weights;
#else
  //resize image
  std::string resize_img = "resized.jpg";
#ifdef USE_OPENCV
  resize_image(FLAGS_input, resize_img, resize_width, resize_height);
#endif
  //get datum
  caffe::Datum datum;
  if (!ReadImageToDatum(resize_img, 1, resize_width, resize_height, &datum)) {
	  LOG(ERROR) << "Error during file reading";
  }

  //get the blob
  Blob<float>* blob = new Blob<float>(1, datum.channels(), datum.height(), datum.width());

  //get the blobproto
  caffe::BlobProto blob_proto;
  blob_proto.set_num(1);
  blob_proto.set_channels(datum.channels());
  blob_proto.set_height(datum.height());
  blob_proto.set_width(datum.width());
  const int data_size = datum.channels() * datum.height() * datum.width();
  int size_in_datum = std::max<int>(datum.data().size(),
	  datum.float_data_size());
  for (int i = 0; i < size_in_datum; ++i) {
	  blob_proto.add_data(0.);
  }
  const string& data = datum.data();
  if (data.size() != 0) {
	  for (int i = 0; i < size_in_datum; ++i) {
		  blob_proto.set_data(i, blob_proto.data(i) + (uint8_t)data[i]);
	  }
  }

  //set data into blob
  blob->FromProto(blob_proto);

  //fill the vector
  vector<Blob<float>*> bottom;
  bottom.push_back(blob);
  float type = 0.0;

  const vector<Blob<float>*>& result = caffe_net.Forward(bottom, &type);
#if 0
  if (FLAGS_version == 2)
  {
	  //add the post-processing for yolo v2
	  int new_w = 0;
	  int new_h = 0;
	  if (((float)resize_width/ side) < ((float)resize_height / side)) {
		  new_w = resize_width;
		  new_h = resize_width;
	  }
	  else {
		  new_h = resize_height;
		  new_w = resize_height;
	  }
	  for (int i = 0; i < n; ++i){
		  box b = boxes[i];
		  b.x = (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw);
		  b.y = (b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth);
		  b.w *= (float)netw / new_w;
		  b.h *= (float)neth / new_h;
		  if (!relative){
			  b.x *= w;
			  b.w *= w;
			  b.y *= h;
			  b.h *= h;
		  }
		  boxes[i] = b;
	  }
  }
#endif

  std::cout << "OK" << std::endl;
#endif
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
  
  return test_detection();
}
