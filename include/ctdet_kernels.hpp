#pragma once

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>


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



/*
 * 1. read image first
 * 2. allocate the gpu mem. of gpu_mat & inp_img
 * 3. affineTransform
 * 4. BGR HWC -> RGB CHW & normalization
 */ 
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
