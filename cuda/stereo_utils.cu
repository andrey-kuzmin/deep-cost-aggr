#include <stdio.h>
#include <assert.h>
#include <math_constants.h>
#include <stdint.h>
#include <unistd.h>

__global__ void census(float *x0, float *x1, float *output, int size, int num_channels, int size2, int size3, int wnd_half, float bnd_const)
{
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int id = blockId * blockDim.x + threadIdx.x;

	if (id < size) {
		
		int x = blockIdx.x;
		int y = blockIdx.y;
		int d = -threadIdx.x;

		float dist;
		if (0 <= x + d && x + d < size3) {
			dist = 0;
			for (int i = 0; i < num_channels; i++) {
				int ind_p = (i * size2 + y) * size3 + x;
				for (int yy = y - wnd_half; yy <= y + wnd_half; yy++) {
					for (int xx = x - wnd_half; xx <= x + wnd_half; xx++) {
						if (0 <= xx && xx < size3 && 0 <= xx + d && xx + d < size3 && 0 <= yy && yy < size2) {
							int ind_q = (i * size2 + yy) * size3 + xx;
							if ((x0[ind_q] < x0[ind_p]) != (x1[ind_q + d] < x1[ind_p + d])) {
								dist++;
							}
						} else {
							dist++;
						}
					}
				}
			}
			dist /= num_channels;
		} else {
			dist = bnd_const;
		}
		output[id] = dist;
	}
}

__global__ void sad_color(float *x0, float *x1, float *output, int size, int size2, int size3, int wnd_half, float bnd_const)
{
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int id = blockId * blockDim.x + threadIdx.x;
    
    int num_channels = 1;
    
	if (id < size) {
		int x = blockIdx.x;
		int y = blockIdx.y;
		int d = -threadIdx.x;

		float dist;
		if (0 <= x + d && x + d < size3) {
			dist = 0;
			for (int i = 0; i < num_channels; i++) {
				int ind_p = (i * size2 + y) * size3 + x;
				for (int yy = y - wnd_half; yy <= y + wnd_half; yy++) {
					for (int xx = x - wnd_half; xx <= x + wnd_half; xx++) {
	         			if (0 <= xx && xx < size3 && 0 <= xx + d && xx + d < size3 && 0 <= yy && yy < size2) {
			    				int ind_q = (i * size2 + yy) * size3 + xx;
    			    		    dist += abs(x0[ind_p] - x1[ind_q + d]);
    			    	}
					}
				}
			}
		} else {
			dist = bnd_const;
		}
		output[id] = dist;
	}
}

__global__ void linear_comb(float *inp0, float *inp1, float *output, int size, float alpha, float beta)
{
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int id = blockId * blockDim.x + threadIdx.x;
    
    if (id < size) {
        output[id] = alpha * inp0[id] + beta * inp1[id];
    } 
    
    __syncthreads();
}

__global__ void outlier_detection(float *d0, float *d1, float *outlier, int size, int dim3, int disp_max)
{
	int id = blockIdx.y * gridDim.x + blockIdx.x;
	if (id < size) {
		int x = id % dim3;
		int d0i = d0[id];
		if (x - d0i < 0) {
			//assert(0);
			outlier[id] = 1;
		} else if (abs(d0[id] - d1[id - d0i]) < 1.1) {
			outlier[id] = 0; /* match */
		} else {
			outlier[id] = 1; /* occlusion */
			for (int d = 0; d < disp_max; d++) {
				if (x - d >= 0 && abs(d - d1[id - d]) < 1.1) {
					outlier[id] = 2; /* mismatch */
					break;
				}
			}
		}
	}
}

__device__ void sort(float *x, int n)
{
	for (int i = 0; i < n - 1; i++) {
		int min = i;
		for (int j = i + 1; j < n; j++) {
			if (x[j] < x[min]) {
				min = j;
			}
		}
		float tmp = x[min];
		x[min] = x[i];
		x[i] = tmp;
	}
}

__global__ void interpolate_mismatch(float *d0, float *outlier, float *out, int size, int dim2, int dim3)
{
	const float dir[] = {
		0	,  1,
		-0.5,  1,
		-1	,  1,
		-1	,  0.5,
		-1	,  0,
		-1	, -0.5,
		-1	, -1,
		-0.5, -1,
		0	, -1,
		0.5 , -1,
		1	, -1,
		1	, -0.5,
		1	,  0,
		1	,  0.5,
		1	,  1,
		0.5 ,  1
	};

	int id = blockIdx.y * gridDim.x + blockIdx.x;
	if (id < size) {
		if (outlier[id] != 2) {
			out[id] = d0[id];
			return;
		}

		float vals[16];
		int vals_size = 0;

		int x = id % dim3;
		int y = id / dim3;
		for (int d = 0; d < 16; d++) {
			float dx = dir[2 * d];
			float dy = dir[2 * d + 1];
			float xx = x;
			float yy = y;
			int xx_i = round(xx);
			int yy_i = round(yy);
			while (0 <= yy_i && yy_i < dim2 && 0 <= xx_i && xx_i < dim3 && outlier[yy_i * dim3 + xx_i] == 2) {
				xx += dx;
				yy += dy;
				xx_i = round(xx);
				yy_i = round(yy);
			}

			int ind = yy_i * dim3 + xx_i;
			if (0 <= yy_i && yy_i < dim2 && 0 <= xx_i && xx_i < dim3) {
				assert(outlier[ind] != 2);
				vals[vals_size++] = d0[ind];
			}
		}
		assert(vals_size > 0);
		sort(vals, vals_size);
		out[id] = vals[vals_size / 2];
	}
}

__global__ void interpolate_occlusion(float *d0, float *outlier, float *out, int size, int dim3)
{
	int id = blockIdx.y * gridDim.x + blockIdx.x;
	if (id < size) {
		if (outlier[id] != 1) {
			out[id] = d0[id];
			return;
		}
		int x = id % dim3;

		int dx = 0;
		while (x + dx >= 0 && outlier[id + dx] != 0) {
		    dx--;
		}
		if (x + dx < 0) {
			dx = 0;
			while (x + dx < dim3 && outlier[id + dx] != 0) {
				dx++;
			}
		}
		if (x + dx < dim3) {
			out[id] = d0[id + dx];
		} else {
			out[id] = d0[id];
		}
	}
}

__global__ void dtransform_lr(
    float* output, float* weight,
    const int height, const int width, const int channels) 
{
  //id_e = (y * width + x) * channels + z;
  //id_w = y * w + x
  
  int ind = 0;
  int ind_prev = 0;
  
  float omega = 0.0;
  
  int i_w = 0;
  for (i_w = 1; i_w < width; i_w++)
  {
      ind = (blockIdx.x * width + i_w) * channels + threadIdx.x;
      ind_prev = (blockIdx.x * width + i_w - 1) * channels + threadIdx.x;
      
      omega = weight[blockIdx.x * width + i_w];
      
      output[ind] = (1.0 - omega) * output[ind] + omega * output[ind_prev];
      
  }
  
  for (i_w = width-2; i_w >= 0; i_w--)
  {
      ind = (blockIdx.x * width + i_w) * channels + threadIdx.x;
      ind_prev = (blockIdx.x * width + i_w + 1) * channels + threadIdx.x;
      
      omega = weight[blockIdx.x * width + i_w];
      
      output[ind] = (1.0 - omega) * output[ind] + omega * output[ind_prev];
  }
}

__global__ void dtransform_ud(
    float* output, float* weight,
    const int height, const int width, const int channels) 
{
  //id_e = (y * width + x) * channels + z;
  //id_w = y * w + x
  
  int ind = 0;
  int ind_prev = 0;
  
  float omega = 0.0;
  
  int i_h = 0;
  for (i_h = 1; i_h < height; i_h++)
  {
      ind = (i_h * width + blockIdx.x) * channels + threadIdx.x;
      ind_prev = ((i_h-1) * width + blockIdx.x) * channels + threadIdx.x;
      
      omega = weight[i_h * width + blockIdx.x];
      
      output[ind] = (1.0 - omega) * output[ind] + omega * output[ind_prev];
      
  }
  
  for (i_h = height-2; i_h >= 0; i_h--)
  {
      ind = (i_h * width + blockIdx.x) * channels + threadIdx.x;
      ind_prev = ((i_h+1) * width + blockIdx.x) * channels + threadIdx.x;
      
      omega = weight[i_h * width + blockIdx.x];
      
      output[ind] = (1.0 - omega) * output[ind] + omega * output[ind_prev];
  }
}





