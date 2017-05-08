int cuda_test(void);
void cuda_segmentation(float *img, int *parents, float sigma, float lambda, 
                       int tilesize, int tau, int img_width, int img_height,
                       double& pdensityTime, double& segmentTreeTime, double&
                       segmentTime);
