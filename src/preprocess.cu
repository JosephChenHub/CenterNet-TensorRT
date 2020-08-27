#include "process.hpp"


/// warpAffine: src -> dst, since the traversal locates on dst rather than src,
/// we need the inverse transformation matrix
/// dst = trans * src, src = inv_trans * dst  

/// src: NHWC, dst:NCHW, keep BGR format
template <typename T>
__global__ void centernet_preprocess_kernel(const T* src,  
        const int batch_size, const int channel, 
        const int in_h, const int in_w, 
        float* dst, const int out_h, const int out_w, 
        const float* inv_trans,
        const float* mean, const bool mean_valid,
        const float* std,  const bool std_valid) {

    const int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t threadY = threadIdx.y + blockIdx.y * blockDim.y;
    if (xIndex >= out_w || threadY >= out_h * batch_size) return;
    const int yIndex = threadY % out_h; 
    const int batch_id = threadY / out_h; 
    const size_t out_pannel = out_h * out_w * channel;
    const size_t in_pannel = in_h * in_w * channel;

    /// calculate the indices of src.
    const float sx = inv_trans[0] * static_cast<float>(xIndex) + inv_trans[1] * static_cast<float>(yIndex) + inv_trans[2];
    const float sy = inv_trans[3] * static_cast<float>(xIndex) + inv_trans[4] * static_cast<float>(yIndex) + inv_trans[5];

    float val = 0.;

    const int h1 = sy;
    const int h1p = (h1 < in_h - 1) ? 1 : 0;

    const float h1lambda = sy - h1 * 1.0;
    const float h0lambda = 1.0  - h1lambda;
    //
    const int w1 = sx;
    const int w1p = (w1 < in_w - 1) ? 1 : 0;

    const float w1lambda = sx - w1 ;
    const float w0lambda = 1.0 - w1lambda;
    const float d_mean[3] = {0.408, 0.447, 0.470};
    const float d_std[3]  = {0.289, 0.274, 0.278};
    //
    for (int c = 0; c < channel; ++c) {
        size_t outIdx = c * out_h * out_w + yIndex * out_w + xIndex + batch_id * out_pannel; // NCHW
        if (sx < 0 || sx > in_w || sy < 0|| sy > in_h) {
            val = 0;
        } else { //! bilinear interpolation 
            val = h0lambda * (w0lambda * src[batch_id*in_pannel + h1 * in_w * channel + w1 * channel + c] + 
                 w1lambda * src[batch_id* in_pannel + h1 * in_w * channel + (w1 + w1p) * channel +  c]) +
                h1lambda * (w0lambda * src[batch_id * in_pannel + (h1 + h1p) * in_w * channel + w1 * channel + c] +
                 w1lambda * src[batch_id * in_pannel + (h1 + h1p) * in_w * channel + (w1 + w1p) * channel + c]);
        }

        dst[outIdx] = (val / 255.0 - d_mean[c] ) / d_std[c];
    } 
}






template <typename T>
void cuda_centernet_preprocess(const T* src, 
        const int batch_size, const int channel, const int in_h, const int in_w,  
        float* dst, const int out_h, const int out_w, 
        const float* inv_trans,  
        const float* mean, const bool mean_valid,
        const float* std, const bool std_valid, 
        cudaStream_t stream) {

    dim3 block(16, 16);
    int grid_x = (out_w + block.x - 1) / block.x;
    int grid_y = (out_h*batch_size + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);

    centernet_preprocess_kernel<T><<<grid, block, 0, stream>>>(src, 
	    batch_size,  channel, in_h, in_w, 
	    dst, out_h, out_w, 
            inv_trans, mean, mean_valid, std, std_valid);

}

template void cuda_centernet_preprocess(const uint8_t*, const int, const int, const int, const int, \
        float*, const int, const int, const float*, const float*, const bool, \
        const float*, const bool, cudaStream_t);


