#pragma once

#include <cuda_runtime_api.h>



template <typename T>
void cuda_centernet_preprocess(const T* src, 
        const int batch_size, const int channel, const int in_h, const int in_w,  
        float* dst, const int out_h, const int out_w, 
        const float* inv_trans,  
        const float* mean, const bool mean_valid,
        const float* std, const bool std_valid, 
        cudaStream_t stream); 


// end of this file
