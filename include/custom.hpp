#pragma once 

#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>




void log2_series(unsigned int value, std::vector<int>& param);
int log2_32 (unsigned int value);

void get_affine_transform(
        float* mat_data, 
        const float* const center, \
        const float* const scale, \
        const float* const shift, 
        const float rot, \
        const int output_h, const int output_w, 
        const bool inv=false);


template <typename T1, typename T2>
void cuda_warp_affine(const int batch_num, 
        T1* src, const int channel, const int in_h, const int in_w,  
        T2* dst, const int out_h, const int out_w, 
        const float* trans, cudaStream_t stream);
