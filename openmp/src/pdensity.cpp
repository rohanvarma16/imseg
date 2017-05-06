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

using namespace cv;



void compute_pdensity(cv::Mat& img_or, std::vector<float> &pdensity,float sigma, int furthest_nbr,float lambda){
pdensity[0]= 100.0;
// convert image to float:
cv::Mat img;
img_or.convertTo(img, CV_32FC1); 

float A_ij,xy_dist,xy_dist_sq,rgb_dist_sq;
float my_r,my_g,my_b,nbr_r,nbr_g,nbr_b;

int top_i, top_j;
  for(int i = 0 ; i < img.rows; i++){
    for(int j = 0 ; j < img.cols ; j++){
      int width = (2 * furthest_nbr) + 1;
      int height = (2 * furthest_nbr) + 1;

      // create rectangle (square) around pixel of side length (2*furthest_nbr) centered
      // pixel (i,j):
      top_i = max(0,i-furthest_nbr);
      top_j = max(0,j-furthest_nbr);

      if(height+top_i  >= img.rows ){
        height = img.rows - top_i;
      }

      if(width+top_j  >= img.cols){
        width =img.cols - top_j;
      }

      cv::Rect myROI( top_j,top_i, width,height);

      cv::Mat nbr_or = img_or(myROI);
      cv::Mat nbr;
      nbr_or.convertTo(nbr,CV_32FC1);

      A_ij = 0.f;
      Point3_<float>* my_p = img.ptr<Point3_<float> >(i,j);
      my_r = my_p->z;
      my_g = my_p->y;
      my_b = my_p->x;

      if(height != nbr.rows || width != nbr.cols){
         printf("height: %d, width: %d, nbr.rows: %d, nbr.cols: %d\n",height,width,nbr.rows,nbr.cols);
}

      for(int i_nbr = 0 ; i_nbr < height ; i_nbr++){
        for(int j_nbr = 0 ; j_nbr < width ; j_nbr++){
          // using Manhattan distance (can use euclidean (??))

          xy_dist = (float) (abs(i_nbr - abs(i-top_i)) + abs(j_nbr - abs(j - top_j)));
          xy_dist_sq = xy_dist * xy_dist;
          Point3_<float>* nbr_p = nbr.ptr<Point3_<float> >(i_nbr,j_nbr);
          nbr_r = nbr_p->z;
          nbr_g = nbr_p->y;
          nbr_b = nbr_p->x;
          rgb_dist_sq = pow(my_r-nbr_r,2.0) + pow(my_g-nbr_g,2.0) + pow(my_b-nbr_b,2.0);
          rgb_dist_sq = lambda * rgb_dist_sq;

          A_ij += exp(-1.f * (pow(xy_dist,2.0) + rgb_dist_sq)/(2.0 * sigma * sigma));
        }
    }

    pdensity[i * img.cols + j] = A_ij;
  }

  }
}