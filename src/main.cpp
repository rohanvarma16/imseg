#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <math.h>      
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <vector> 
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "../include/pdensity.h"
#include "../include/segmentation.h"
#include <set>
#include <iostream>
#include <unistd.h>
#include "../include/cycleTimer.h"

using namespace cv;
using namespace std;
void showhelpinfo(char *s)
{
  cout<<"Usage:   "<<s<<" [-option] [argument]"<<endl;
  cout<<"option:  "<<"-h  show help information"<<endl;
  cout<<"         "<<"-f path to image"<<endl;
  cout<<"         "<<"-d  neighborhood size for density computation"<<endl;
  cout<<"         "<<"-t  cluster size tuning parameter "<<endl;
}

int main(int argc, char** argv )
{
  // top level parameters:

  //default:
  int furthest_nbr = 8; 
  int tau = 10; 

    if(argc == 1)
  {
    showhelpinfo(argv[0]);
    exit(1);
  }

  char tmp;
  char* imgpath;
  while((tmp=getopt(argc,argv,"hf:d:t:"))!=-1)
  {
    switch(tmp)
    {
      /*option h show the help infomation*/
      case 'h':
        showhelpinfo(argv[0]);
        return(1);
        break;
      /*option u present the username*/
      case 'f':
        cout<<" Image is "<<optarg<<endl;
        imgpath = optarg;
        break;
      /*option p present the password*/ 
      case 'd':
        cout<<"Furthest nbr dist is "<<optarg<<endl;
        furthest_nbr = atoi(optarg);
        break;
        case 't':
        cout<<"Cluster size parameter tau is "<<optarg<<endl;
        tau = atoi(optarg);
        break;
      default:
        showhelpinfo(argv[0]);
      break;
    }
  }

  
  double readTime = 0.f;
  double pdensityTime = 0.f;
  double segmentTreeTime = 0.f;
  double segmentTime = 0.f;
  double writeTime = 0.f;
  double totalTime = 0.f;

  double startTime = CycleTimer::currentSeconds();

  cv::Mat img = cv::imread(imgpath,CV_LOAD_IMAGE_COLOR);
    if(! img.data )                              // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

  printf("img rows: %d, img cols: %d\n",img.rows,img.cols);

  double endReadTime = CycleTimer::currentSeconds();

  /************** STEP 1: COMPUTE P_DENSITY ****************/

  std::vector<float> pdensity(img.rows * img.cols , 0.0);

  float sigma = ((float) furthest_nbr)/3.f;
  float lambda = 0.01;
  compute_pdensity(img,pdensity,sigma,furthest_nbr,lambda);
  printf("computed density\n");

  double endPdensityTime = CycleTimer::currentSeconds();

  /************** STEP 2: COMPUTE PARENTS ****************/

  std::vector<int> parents(img.rows * img.cols);
  std::vector<float> distances(img.rows * img.cols , 0.0);

  segmentTree(img, parents, distances, pdensity, tau);
  printf("found parents\n");
  
  double endSegmentTreeTime = CycleTimer::currentSeconds();

  constructSegments(img,parents,distances);
  printf("constructed segments, ");
  //printf("num_segments: %d \n", distinct_abs(parents));

  double endSegmentTime = CycleTimer::currentSeconds();

  /************** STEP 3: VISUALIZE SEGMENTS **************/

  cv::Mat img_seg = img.clone();
  constructSegmentedImg(img,img_seg, parents);

  double endWriteTime = CycleTimer::currentSeconds();
  
  readTime = 1000.f * (endReadTime - startTime);  
  pdensityTime = 1000.f * (endPdensityTime - endReadTime);
  segmentTreeTime = 1000.f * (endSegmentTreeTime - endPdensityTime);
  segmentTime = 1000.f * (endSegmentTime - endSegmentTreeTime);
  writeTime = 1000.f * (endWriteTime - endSegmentTime);
  totalTime = 1000.f * (endWriteTime - startTime);
 
  printf("Time for reading input image file:      %.4f ms\n", readTime);
  printf("Time for calculating point densities:   %.4f ms\n", pdensityTime);
  printf("Time for constructing segment tree:     %.4f ms\n", segmentTreeTime);
  printf("Time for constructing segments:         %.4f ms\n", segmentTime);
  printf("Time for writing the output image:      %.4f ms\n", writeTime);
  printf("Overall time:                           %.4f ms\n", totalTime);

  // Create big mat for window
  cv::Mat win_mat(cv::Size(2 * img.cols , img.rows), CV_8UC3);
 
  // Copy small images into big mat
  img.copyTo(win_mat(cv::Rect(  0, 0, img.cols, img.rows)));
  img_seg.copyTo(win_mat(cv::Rect(img.cols, 0, img.cols, img.rows)));
 
  // Display big mat
  cv::imshow("Original and Segmented", win_mat);
  waitKey(0); 

  return 0;
}
