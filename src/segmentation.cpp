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
#include <vector> 

using namespace cv;


void segmentTree(cv::Mat& img_or, std::vector<int>& parents, std::vector<float>& distances,std::vector<float>& pdensity,int furthest_nbr ){

// convert image to float:
cv::Mat img;
img_or.convertTo(img, CV_32FC1); 

int top_i, top_j;
float nbr_pdensity;
float my_pdensity;
int my_parent;
float xy_dist;
int counter = 0;
  for(int i = 0 ; i < img.rows; i++){
    for(int j = 0 ; j < img.cols ; j++){

      int width = (2 * furthest_nbr) + 1;
      int height = (2 * furthest_nbr) + 1;
      // create rectangle (square) around pixel of side length (2*furthest_nbr) centered
      // pixel (i,j):
     // printf("i: %d, j: %d \n",i,j);
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


      float min_dist = 10000000;
      my_pdensity = pdensity[i * img.cols + j];
      my_parent = i * img.cols + j;
      //iterate over neighborhood:
     
      for(int i_nbr = 0 ; i_nbr < height ; i_nbr++)
        for(int j_nbr = 0 ; j_nbr < width ; j_nbr++){
          // using Manhattan distance (can use euclidean (??))
        	int nbr_row = top_i + i_nbr;
          	int nbr_col = top_j + j_nbr;
          	int nbr_ind = (nbr_row )* img.cols + nbr_col;
        	nbr_pdensity = pdensity[nbr_ind];

          xy_dist = (float) (abs(i_nbr - abs(i-top_i)) + abs(j_nbr - abs(j - top_j)));
          
          if(i== nbr_row && j == nbr_col){
          	counter++;
          	continue;
          }

          if( nbr_pdensity > my_pdensity && xy_dist < min_dist ){
          	min_dist = xy_dist;
          

          	if(nbr_row >= img.rows || nbr_col >= img.cols){
          		printf("top_i: %d, top_j: %d, i_nbr: %d, j_nbr: %d,nbr_row:%d,nbr_col:%d, height: %d, width: %d \n ",top_i,top_j,i_nbr,j_nbr, nbr_row,nbr_col,height,width);
          	}
          	my_parent = nbr_ind;
          }
        }
        distances[i * img.cols + j] = min_dist;
        parents[i * img.cols + j] = my_parent;
    }
  }
  printf("counter: %d \n",counter);
}




void constructSegments(cv::Mat &img, std::vector<int>& parents, std::vector<float>&distances){

int num_pts = img.cols * img.rows;

  int delta;
  int done = 0;
  int treedepth= 0;


  while(! done)
    { delta =0;

      for(int i =0 ; i< num_pts ; i++){

        if(parents[parents[i]] != parents[i]){
          parents[i] = parents[parents[i]];
          delta++;
        }
      }
      if(delta == 0){
        done = 1;
      }
      treedepth++;
    }
    printf("treedepth: %d \n",treedepth);

  }



  void constructSegmentedImg(cv::Mat& img,cv::Mat& img_seg,std::vector<int>& parents){

    std::vector<int> clusterids;
    int num_pixels = img.rows * img.cols;

    for(int i = 0; i < num_pixels ; i++){
      if (parents[i]==i)
      {
        clusterids.push_back(i);
      }
    }

    printf("created clusterIDs \n");

    std::map< int,std::vector<int> > parent_map;
    //construct segments:
    for (std::vector<int>::iterator it = clusterids.begin(); it != clusterids.end(); ++it)
    {
      parent_map.insert(std::pair<int,std::vector<int> >(*it,std::vector<int>()));
    }

    for (int i = 0; i < num_pixels; ++i)
    {
      parent_map[parents[i]].push_back(i);
    } 

    printf("created map between clusterIds and members \n ");

    std::map<int,int> red_map;
    std::map<int,int> green_map;
    std::map<int,int> blue_map;

    // iterate over each segment:
    for (std::vector<int>::iterator i = clusterids.begin(); i != clusterids.end(); ++i)
    { 
      int red_sum = 0;
      int green_sum = 0;
      int blue_sum = 0;
      int count = 0;
    // iterate over each point in segment
      for (std::vector<int>::iterator j = parent_map[*i].begin(); j != parent_map[*i].end(); ++j)
      {
        int pixel_ind = *j;
        int pixel_row = pixel_ind/img.cols;
        int pixel_col = pixel_ind % img.cols;
        Vec3b & color = img.at<Vec3b>(pixel_row,pixel_col);

        int red = static_cast< int >(color[2]);
        int green = static_cast< int >(color[1]);
        int blue = static_cast< int >(color[0]);

        blue_sum += blue;
        green_sum += green;
        red_sum += red;
        count++;
      }

      red_sum =  red_sum/count;
      green_sum =  green_sum/count;
      blue_sum = blue_sum/count;
      red_map[*i] = red_sum;
      green_map[*i] = green_sum;
      blue_map[*i] = blue_sum;
    } 

    printf("computed color composition for each segment (avg for viz) \n ");
    

    for (std::vector<int>::iterator i = clusterids.begin(); i != clusterids.end(); ++i){

      uint8_t r = (uint8_t) red_map[*i];
      uint8_t g = (uint8_t) green_map[*i];
      uint8_t b = (uint8_t) blue_map[*i];

      for (std::vector<int>::iterator j = parent_map[*i].begin(); j != parent_map[*i].end(); ++j){
        int pixel_ind = *j;
         int pixel_row = pixel_ind/img.cols;
        int pixel_col = pixel_ind % img.cols;
        Vec3b & color = img_seg.at<Vec3b>(pixel_row,pixel_col);
        color[0] = b;
        color[1] = g;
        color[2] = r;
        
      }
    }
    printf("constructed segmented image\n");
  }



  
/*int distinct_abs(const vector<int>& v)
{
   std::set<int> distinct_container;

   for(auto curr_int = v.begin(), end = v.end(); // no need to call v.end() multiple times
       curr_int != end;
       ++curr_int)
   {
       // std::set only allows single entries
       // since that is what we want, we don't care that this fails 
       // if the second (or more) of the same value is attempted to 
       // be inserted.
       distinct_container.insert(*curr_int);
   }

   return distinct_container.size();
}*/





