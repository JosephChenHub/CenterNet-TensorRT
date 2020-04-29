/*
 * This file contains some kernels including the pre-processing, post-processing of ctdet. 
 * 
 */
#include <math.h>
#include "det_kernels.hpp"
#include <iostream>

#include "gpu_common.cuh"
#include "custom.hpp"

//using namespace std;



template <typename T>
__device__ __forceinline__ static T area_pixel_compute_source_index(T scale,
                                     int dst_index,
                                     bool align_corners,
                                     bool cubic = false) {
    if (align_corners) {
        return scale * dst_index;
    }
    else {
        T src_idx = scale * (dst_index + static_cast<T>(0.5)) - static_cast<T>(0.5);
        return (!cubic && src_idx < static_cast<T>(0)) ? static_cast<T>(0): src_idx;
    }
}

template <typename T>
__host__ __forceinline__ T area_pixel_compute_scale(int input_size, int output_size, bool align_corners) {
    if(output_size > 1) {
        return align_corners ? static_cast<T>(input_size - 1) / (output_size - 1) : static_cast<T>(input_size) / output_size;
    }
    else {
        return static_cast<T>(0);
    }
}


/*
 * in_img: NHWC, out_img: NCHW (may be normalized), mean/std: gpu mem. 
 *  C == 3 
 */ 
__global__ void preprocess_kernel(const int batch_size, 
        const uint8_t* in_img, const int channels, const int in_h, const int in_w, 
        float* out_img, const int out_h, const int out_w, 
        const float rheight, const float rwidth, const bool align_corners, 
        const float padding_val, 
        const ScaleOp type, 
        const float* mean, const bool mean_valid, 
        const float* std, const bool std_valid) {

    /// 2D Index of current thread
    size_t threadX = blockIdx.x * blockDim.x + threadIdx.x;
    size_t threadY = blockIdx.y * blockDim.y + threadIdx.y;
    size_t batch_id = static_cast<int>(threadY / out_h);
    int yIndex = threadY % out_h; 
    int xIndex = threadX;
    if(xIndex >= out_w || threadY >= out_h * batch_size) return;
    const size_t out_pannel = out_h * out_w * channels;  


    if (in_h != out_h || in_w != out_w) {  
        if (type == ScaleOp::Resize) { //! resize using bilinear interpolation 
            float h1r = area_pixel_compute_source_index<float>(rheight, yIndex, align_corners, /*cubic=*/false);
            const int h1 = h1r;
            const int h1p = (h1 < in_h - 1) ? 1 : 0;
            const float h1lambda = h1r - h1;
            const float h0lambda = static_cast<float>(1) - h1lambda;
            //
            float w1r = area_pixel_compute_source_index<float>(rwidth, xIndex, align_corners, /*cubic=*/false);
            const int w1 = w1r;
            const int w1p = (w1 < in_w - 1) ? 1 : 0;
            const float w1lambda = w1r - w1;
            const float w0lambda = static_cast<float>(1) - w1lambda;
            //
            for (int c = 0; c < channels; ++c) {
                const int outIdx = c * out_h * out_w + yIndex * out_w + xIndex + batch_id * out_pannel; // NCHW
                const float val = 
                    h0lambda * (w0lambda * in_img[batch_id*out_pannel + h1 * in_w * channels + w1 * channels + channels - 1 - c] +
                     w1lambda * in_img[batch_id*out_pannel + h1 * in_w * channels + (w1 + w1p) * channels +  channels - 1 - c]) +
                    h1lambda *
                    (w0lambda * in_img[batch_id * out_pannel + (h1 + h1p) * in_w * channels + w1 * channels + channels - 1 - c] +
                     w1lambda * in_img[batch_id * out_pannel + (h1 + h1p) * in_w * channels + (w1 + w1p) * channels + channels -1 -c]);
                out_img[outIdx] = val / 255.0; 
               if(mean_valid) out_img[outIdx] -= mean[c];
               if(std_valid)  out_img[outIdx] /= std[c];
            } 
        } else if (type == ScaleOp::Padding) { //! padding, center aligned 
            const int x1_x = out_w > in_w ? (out_w - in_w) / 2 : 0;
            const int x1_y = out_h > in_h ? (out_h - in_h) / 2 : 0;
            for(int c = 0; c < channels; ++c) {
                const int outIdx = c * out_h * out_w + yIndex * out_w + xIndex + batch_id * out_pannel;
                if (yIndex >= x1_y && yIndex < x1_y + in_h && 
                        xIndex >= x1_x && xIndex < x1_x + in_w) {
                    const int inIdx = (yIndex - x1_y) * in_w * channels + (xIndex - x1_x) * channels + batch_id * out_pannel; 
                    out_img[outIdx] = static_cast<float>(in_img[inIdx + channels - 1 - c]) / 255.0; // BGR -> RGB
                    if(mean_valid) out_img[outIdx] -= mean[c];
                    if(std_valid)  out_img[outIdx] /= std[c];
                } else {
                    out_img[outIdx] = padding_val;  
                }
            }
        }
    } else { //! same shape
        const int idx = batch_id * out_pannel +  
                yIndex * in_w * channels + xIndex * channels; // NHWC
        for(int c = 0; c < channels; ++c) {
            const int outIdx = c * out_h * out_w + yIndex * out_w + \
			       xIndex + batch_id * out_pannel; //NCHW
            //out_img[outIdx] = static_cast<float>(in_img[idx + channels - 1 - c]) / 255.0; // BGR -> RGB
            out_img[outIdx] = static_cast<float>(in_img[idx + c]) / 255.0;  // keep BGR format
            if(mean_valid) out_img[outIdx] -= mean[c];
            if(std_valid)  out_img[outIdx] /= std[c];
        }
    }
}




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
        const float* mean, const bool mean_valid,
        const float* std, const bool std_valid, 
        cudaStream_t& stream) {
    int new_height = static_cast<int>(img.rows * scale);
    int new_width  = static_cast<int>(img.cols * scale);
    int inp_h, inp_w;
    float c[2], s[2];
    if(fix_res) {
        inp_h = input_h;
        inp_w = input_w;
        c[0] = new_width / 2.;
        c[1] = new_height / 2.;
        s[0] = img.rows > img.cols ? img.rows : img.cols;
        s[1] = s[0];
    } else {
        inp_h = (new_height | pad) + 1;
        inp_w = (new_width | pad) + 1;
        c[0] = new_width / 2 ;
        c[1] = new_height / 2;
        s[0] = inp_w;
        s[1] = inp_h;
    }
    /// affine_transform
    float shift[2] = {0., 0.};
    using namespace cv;
    cv::Mat warp_mat (2, 3, CV_32FC1);
    get_affine_transform((float*)warp_mat.data, c, s, shift, 0, inp_h, inp_w);
    get_affine_transform(inv_trans, c, s, shift, 0, inp_h / down_ratio, inp_w / down_ratio, true);

    //std::cout << "warp_mat:\n" << warp_mat << std::endl;
    if (new_width != img.cols && new_height != img.rows) {
        cv::resize(img, img, cv::Size(new_width, new_height));
    }
    //cv::Mat inp_img = cv::Mat::zeros(inp_h, inp_w, img.type());
    warpAffine(img, inp_img, warp_mat, inp_img.size());
    //cv::imwrite("warp2.png", inp_img);
    /// copy to gpu memory 
    CHECK_CUDA(cudaMemcpyAsync((void*)gpu_mat, inp_img.data, sizeof(uint8_t)*inp_img.rows*inp_img.cols*3, cudaMemcpyHostToDevice, stream));
    /// BGR HWC -> RGB, CHW & normalization
    dim3 block(16, 16);
    int grid_x = (inp_h + block.x - 1) / block.x;
    int grid_y = (inp_w*batch_size + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);
    // bool align_corners = true;
    // float rheight = area_pixel_compute_scale<float>(inp_h, inp_h, align_corners);
    // float rwidth  = area_pixel_compute_scale<float>(inp_w, inp_w, align_corners);

    preprocess_kernel<<<grid, block, 0, stream>>>(
	        batch_size, 
            static_cast<const uint8_t* >(gpu_mat), 
            3, inp_h, inp_w,  
            d_out, inp_h, inp_w, 
            1, 1, true, 
            0, 
            ScaleOp::Same, 
            static_cast<const float* >(mean), mean_valid, 
            static_cast<const float* >(std), std_valid);

    CHECK_LAST_ERR("preprocess_kernel");
}



