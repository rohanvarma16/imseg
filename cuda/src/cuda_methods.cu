#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>
#include <cuda.h>
#include "../include/cycleTimer.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}



#define MAX_TILESIZE 32

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}



int cuda_test(void){
  
  int N = 1<<20;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);

}


__global__
void compute_pdensity_kernel(float *device_img, float *device_pdensity, 
                             float sigma, float lambda, int img_width, 
                             int img_height)
{
  // Y -> ROW  , X -> COLUMN 
  // j -> ROW  , i -> COLUMN 
  int block_i = blockIdx.x;
  int block_j = blockIdx.y;
 
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if((i == 0) && (j == 0))
    printf("pdensity blockdim = %d, griddim = %d\n", blockDim.x, gridDim.x);
  
  // Shared memory buffer to hold image data.
  __shared__ float3 img_rgb[MAX_TILESIZE*MAX_TILESIZE];

  float3 my_rgb, nbr_rgb;
  int nbr_block_i, nbr_block_j, nbr_id, nbr_i, nbr_j;
  float xy_dist, xy_dist_sq, rgb_dist_sq, A_ij = 0;

  // Load my_rgb.
  if((i < img_width) && (j < img_height))
    my_rgb = *(float3 *) &device_img[3*(j*img_width + i)];

  // Iterate over the neighbouring blocks including yourself.
  for(int nbr_block_x = -1; nbr_block_x < 2; nbr_block_x++)
  {
    for(int nbr_block_y = -1; nbr_block_y < 2; nbr_block_y++)
    {
      nbr_block_i = block_i + nbr_block_x;
      nbr_block_j = block_j + nbr_block_y;

      // Skip if block is out of bounds.
      if((nbr_block_i < 0) || (nbr_block_i >= gridDim.x) ||
         (nbr_block_j < 0) || (nbr_block_j >= gridDim.y))
        continue;
    
      // Load block's worth of img data into shared memory.
      // Each thread loads one float3.
      nbr_id = threadIdx.y * blockDim.x + threadIdx.x;
      nbr_i = nbr_block_i * blockDim.x + threadIdx.x;
      nbr_j = nbr_block_j * blockDim.y + threadIdx.y;

      if((nbr_i < img_width) && (nbr_j < img_height))
        img_rgb[nbr_id] = *(float3*) &device_img[3*(nbr_j*img_width + nbr_i)];

      __syncthreads();

      // Iterate over loaded neighbourhood and compute partial pdensity.
      for(int nbr_x = 0; nbr_x < blockDim.x; nbr_x++)
      {
        for(int nbr_y = 0; nbr_y < blockDim.y; nbr_y++)
        {
          nbr_id = nbr_y * blockDim.x + nbr_x;
          nbr_i = nbr_block_i * blockDim.x + nbr_x;
          nbr_j = nbr_block_j * blockDim.y + nbr_y;
          
          if((i >= img_width) || (j >= img_height) ||
             (nbr_i >= img_width) || (nbr_j >= img_height))
            continue;

          nbr_rgb = img_rgb[nbr_id];
           
          xy_dist = (float) (abs(i - nbr_i) + abs(j-nbr_j));
          xy_dist_sq = xy_dist * xy_dist;
          
          rgb_dist_sq = pow(my_rgb.x-nbr_rgb.x, 2.0) + pow(my_rgb.y-nbr_rgb.y, 2.0) +
                        pow(my_rgb.z-nbr_rgb.z, 2.0);
          rgb_dist_sq = lambda *rgb_dist_sq;

          A_ij += __expf(-1.0f * ((xy_dist_sq + rgb_dist_sq)/(2.0 * sigma * sigma)));
        }
      }
    
    }
    // One neighbourhood block done.
    __syncthreads();
  }
  
  // Write pdensity.
  if((i < img_width) && (j < img_height))
    device_pdensity[j*img_width + i] = A_ij;

}



__global__
void compute_segment_kernel(float *device_img, float *device_pdensity,
                            int *device_parents, int tau, int img_width,
                            int img_height) 
{
  // Y -> ROW  , X -> COLUMN 
  // j -> ROW  , i -> COLUMN 
  int block_i = blockIdx.x;
  int block_j = blockIdx.y;
 
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if((i == 0) && (j == 0))
    printf("segment blockdim = %d, griddim = %d\n", blockDim.x, gridDim.x);

  // Shared memory buffer to hold pdensity data.
  __shared__ float block_pdensity[MAX_TILESIZE*MAX_TILESIZE];

  int nbr_block_i, nbr_block_j, nbr_id, nbr_i, nbr_j, my_parent;
  float xy_dist, my_pdensity, nbr_pdensity, min_dist = 10000000;

  // Load my_pdensity.
  if((i < img_width) && (j < img_height))
  {
    my_pdensity = device_pdensity[j*img_width + i];
    my_parent = j*img_width + i;
  }

  // Iterate over the neighbouring blocks including yourself.
  for(int nbr_block_x = -1; nbr_block_x < 2; nbr_block_x++)
  {
    for(int nbr_block_y = -1; nbr_block_y < 2; nbr_block_y++)
    {
      nbr_block_i = block_i + nbr_block_x;
      nbr_block_j = block_j + nbr_block_y;

      // Skip if block is out of bounds.
      if((nbr_block_i < 0) || (nbr_block_i >= gridDim.x) ||
         (nbr_block_j < 0) || (nbr_block_j >= gridDim.y))
        continue;
    
      // Load block's worth of pdensity data into shared memory.
      // Each thread loads one float.
      nbr_id = threadIdx.y * blockDim.x + threadIdx.x;
      nbr_i = nbr_block_i * blockDim.x + threadIdx.x;
      nbr_j = nbr_block_j * blockDim.y + threadIdx.y;

      if((nbr_i < img_width) && (nbr_j < img_height))
        block_pdensity[nbr_id] = device_pdensity[nbr_j*img_width + nbr_i];

      __syncthreads();

      // Iterate over loaded neighbourhood and compute partial parent.
      for(int nbr_x = 0; nbr_x < blockDim.x; nbr_x++)
      {
        for(int nbr_y = 0; nbr_y < blockDim.y; nbr_y++)
        {
          nbr_id = nbr_y * blockDim.x + nbr_x;
          nbr_i = nbr_block_i * blockDim.x + nbr_x;
          nbr_j = nbr_block_j * blockDim.y + nbr_y;
          
          if((i >= img_width) || (j >= img_height) ||
             (nbr_i >= img_width) || (nbr_j >= img_height))
            continue;

          nbr_pdensity = block_pdensity[nbr_id];
           
          xy_dist = (float) (abs(i - nbr_i) + abs(j-nbr_j));
          
          if((i == nbr_i) && (j == nbr_j))
            continue;
          
          if((nbr_pdensity > my_pdensity) && (xy_dist < min_dist))
          {
            min_dist = xy_dist;
            my_parent = nbr_j*img_width + nbr_i;
          }
        }
      }
    }
    // One neighbourhood block done.
    __syncthreads();
  }
  
  // Write parent.
  if((i < img_width) && (j < img_height))
    device_parents[j*img_width + i] = my_parent;

}



void cuda_segmentation(float *img, int *parents, float sigma, float lambda, 
                       int tilesize, int tau, int img_width, int img_height,
                       double& pdensityTime, double& segmentTreeTime, double&
                       segmentTime)
{
  float *device_img, *device_pdensity; 
  int *device_parents, *host_parents;

  cudaMalloc(&device_img, img_width*img_height*3*sizeof(float));
  cudaMalloc(&device_pdensity, img_width*img_height*sizeof(float));
  
  thrust::device_vector<int> dev_parents(img_width*img_height); 
  thrust::host_vector<int> h_parents(img_width*img_height);
  thrust::device_vector<int> temp1(img_width*img_height);
  
  device_parents = thrust::raw_pointer_cast(dev_parents.data());
  host_parents = thrust::raw_pointer_cast(h_parents.data());

  gpuErrchk(cudaMemcpy(device_img, img, img_width*img_height*3*sizeof(float), 
                       cudaMemcpyHostToDevice));
  
  /************** STEP 1: COMPUTE P_DENSITY ****************/
  
  int tiles_x = (img_width+tilesize-1) / tilesize;
  int tiles_y = (img_height+tilesize-1) / tilesize;
  
  if(tilesize > MAX_TILESIZE){
    printf("Exceeded maximum neighbourhood distance\n");
    return;
  }

  dim3 gridDim1(tiles_x, tiles_y, 1);
  dim3 blockDim1(tilesize, tilesize, 1);

  double startTime = CycleTimer::currentSeconds();

  compute_pdensity_kernel<<<gridDim1, blockDim1>>>(device_img, device_pdensity, 
                                                 sigma, lambda, img_width, 
                                                 img_height);
  
  cudaDeviceSynchronize();
  double endPdensityTime = CycleTimer::currentSeconds();
  
  printf("Finished cuda pdensity computation.\n");
  /************** STEP 2: COMPUTE PARENTS ****************/
  
  tiles_x = (img_width+tau-1) / tau;
  tiles_y = (img_height+tau-1) / tau;
  
  dim3 gridDim2(tiles_x, tiles_y, 1);
  dim3 blockDim2(tau, tau, 1);

  compute_segment_kernel<<<gridDim2, blockDim2>>>(device_img, device_pdensity, 
                                                device_parents, tau, img_width,
                                                img_height);
  cudaDeviceSynchronize();
  double endSegmentTreeTime = CycleTimer::currentSeconds();
  printf("Finished cuda segment computation.\n");

  int num_gather = 7;

  for(int i = 0; i < num_gather; i++)
  {
    thrust::gather(thrust::device, dev_parents.begin(), dev_parents.end(), 
                   dev_parents.begin(), temp1.begin());
    thrust::copy(temp1.begin(), temp1.end(), dev_parents.begin());
  }

  cudaDeviceSynchronize();
  double endSegmentTime = CycleTimer::currentSeconds();

  pdensityTime = 1000.f * (endPdensityTime - startTime);
  segmentTreeTime = 1000.f * (endSegmentTreeTime - endPdensityTime);
  segmentTime = 1000.f * (endSegmentTime - endSegmentTreeTime);

  // Create big mat for window

  printf("Finished cuda tree cutting.\n");
  thrust::copy(dev_parents.begin(), dev_parents.end(), h_parents.begin());

  memcpy(parents, host_parents, img_width*img_height*sizeof(int));
}









