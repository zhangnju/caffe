// This file was mostly taken from the example given here:
// http://www.votchallenge.net/howto/integration.html

// Uncomment line below if you want to use rectangles
#define VOT_RECTANGLE
//#include "examples/goturn/native/vot.h"

#include "examples/goturn/tracker/tracker.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <io.h>
using namespace std;

DEFINE_string(input, "",
	"The file folder for .");
DEFINE_string(model, "",
	"The model definition protocol buffer text file.");
DEFINE_string(weight, "",
	"The trained caffemodel file for inference or re-train.");
DEFINE_int32(showing, 0,
	"The trained caffemodel file for inference or re-train.");

vot_region GetROI(string FileFolder)
{
	// Read groundtruth for the 1st frame
	vot_region roi;
	if (NULL != FileFolder.c_str())
	{
		ifstream groundtruthFile;
		string groundtruth = "\\groundtruth.txt";
		groundtruthFile.open(FileFolder + groundtruth);
		string firstLine;
		getline(groundtruthFile, firstLine);
		groundtruthFile.close();

		istringstream ss(firstLine);

		// Read groundtruth like a dumb
		float x1, y1, x2, y2, x3, y3, x4, y4;
		char ch;
		ss >> x1; ss >> ch; ss >> y1; ss >> ch; ss >> x2; ss >> ch; ss >> y2; ss >> ch;
		ss >> x3; ss >> ch; ss >> y3; ss >> ch; ss >> x4; ss >> ch; ss >> y4;

		// Using min and max of X and Y for groundtruth rectangle
		float xMin = min(x1, min(x2, min(x3, x4)));
		float yMin = min(y1, min(y2, min(y3, y4)));
		float width = max(x1, max(x2, max(x3, x4))) - xMin;
		float height = max(y1, max(y2, max(y3, y4))) - yMin;

		roi.x = xMin; roi.y = yMin; roi.width = width; roi.height = height;
	}
	return roi;
}

static int  FrameNO=0;
static intptr_t hFile;
static _finddata_t fileinfo;
static bool IsVideoFile = false;
bool GetFrame(string FileFolder, cv::Mat &orig_image)
{
	if (IsVideoFile)
	{
	}
	else
	{
		string jpgfile = FileFolder + "\\*.jpg";
		char filename[_MAX_PATH];
		strcpy(filename, FileFolder.c_str());
		if (FrameNO == 0)
		{
			if ((hFile = _findfirst(jpgfile.c_str(), &fileinfo)) != -1)
			{
				string tmp = "\\";
				strcat(filename, tmp.c_str());
				strcat(filename, fileinfo.name);
			}
			else
			{
				return false;
			}

		}
		else
		{
			if (_findnext(hFile, &fileinfo) == 0)
			{
				string tmp = "\\";
				strcat(filename, tmp.c_str());
				strcat(filename, fileinfo.name);
			}
			else
			{
				_findclose(hFile);
				return false;
			}
		}
		orig_image = cv::imread(filename);
		FrameNO++;
	}

	return true;
}
int main (int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

 // Usage message.
 gflags::SetUsageMessage("Test a object tracking model\n"
		"Usage:\n"
		"    goturn [FLAGS] \n");
 gflags::ParseCommandLineFlags(&argc, &argv, true);

  const string& model_file = FLAGS_model;
  const string& trained_file = FLAGS_weight;

  int gpu_id = 0;
  const bool do_train = false;
  Regressor regressor(model_file, trained_file, gpu_id, do_train);

  // Ensuring randomness for fairness.
#if defined(_WIN32)
  srand( (unsigned)time( NULL ) );
#else
  srandom(time(NULL));
#endif

  // Create a tracker object.
  const bool show_intermediate_output = (bool)FLAGS_showing;
  Tracker tracker(show_intermediate_output);

  //VOT vot; // Initialize the communcation

  // Get region and first frame
  string path = FLAGS_input;
  vot_region region = GetROI(path);
   

  // Load the first frame and use the initialization region to initialize the tracker.
  cv::Mat orig_image;
  if(!GetFrame(path, orig_image))
	  return -1;
  tracker.Init(orig_image, region, &regressor);

  //track
  while (true) {
      // Load current image.
	  if (!GetFrame(path, orig_image))
		  break;
	  const cv::Mat& image = orig_image;

      // Track and estimate the bounding box location.
      BoundingBox bbox_estimate;
      if(!tracker.Track(image, &regressor, &bbox_estimate))
		  break;

      //bbox_estimate.GetRegion(&region);

      //vot.report(region); // Report the position of the tracker
  }

  // Finishing the communication is completed automatically with the destruction
  // of the communication object (if you are using pointers you have to explicitly
  // delete the object).

  return 0;
}
