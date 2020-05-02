#pragma once

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>


#ifdef USE_CV_WARP_AFFINE //! cv::warpAffine is slower than GPU's implementation
enum class ScaleOp {
    Resize,
    Padding,
    Crop,
    Same
};

enum class DimType {
    CHW,
    HWC
};

void cuda_preprocess(const int batch_size,
        float* d_out, 
        const uint8_t* gpu_mat, \
        const cv::Mat& img, 
        cv::Mat& inp_img, 
        const int input_h, const int input_w, 
        const float scale, const int pad, \
        const bool fix_res, const float down_ratio, \
        float* inv_trans,  
        const float* const mean, const bool mean_valid,
        const float* const std, const bool std_valid, 
        cudaStream_t& stream);
#else 
template <typename T>
void cuda_centernet_preprocess(const int batch_num, 
        T* src, const int channel, const int in_h, const int in_w,  
        float* dst, const int out_h, const int out_w, 
        const float* inv_trans,  
        const float* mean, const bool mean_valid,
        const float* std, const bool std_valid, 
        cudaStream_t stream);

#endif 
