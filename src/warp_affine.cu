#include <cuda_runtime_api.h>
#include "custom.hpp"




/// warpAffine: src -> dst, since the traversal locates on dst rather than src,
/// we need the inverse transformation matrix
/// dst = trans * src, src = inv_trans * dst  

template <typename T1, typename T2>
__global__ void warp_affine_kernel(const int batch_num, 
        T1* src,  const int channel, 
        const int in_h, const int in_w, 
        T2* dst, const int out_h, const int out_w, 
        const float* inv_trans) {

    const int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t threadY = threadIdx.y + blockIdx.y * blockDim.y;
    if (xIndex >= out_w || threadY >= out_h * batch_num) return;
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
    size_t outIdx; 
    //
    for (int c = 0; c < channel; ++c) {
        outIdx = c  + yIndex * out_w * channel + xIndex * channel + batch_id * out_pannel; // NHWC
        if (sx < 0 || sx > in_w || sy < 0|| sy > in_h) {
            dst[outIdx] = 0;
        } else { //! bilinear interpolation 
            //const int outIdx = c * out_h * out_w + yIndex * out_w + xIndex + batch_id * out_pannel; // NCHW
            val = h0lambda * (w0lambda * src[batch_id*in_pannel + h1 * in_w * channel + w1 * channel + c] +
                 w1lambda * src[batch_id* in_pannel + h1 * in_w * channel + (w1 + w1p) * channel +  c]) +
                h1lambda * (w0lambda * src[batch_id * in_pannel + (h1 + h1p) * in_w * channel + w1 * channel + c] +
                 w1lambda * src[batch_id * in_pannel + (h1 + h1p) * in_w * channel + (w1 + w1p) * channel + c]);
            
            dst[outIdx] = static_cast<T2>(val); 
        }
    } 
}






template <typename T1, typename T2>
void cuda_warp_affine(const int batch_num, 
        T1* src, const int channel, const int in_h, const int in_w,  
        T2* dst, const int out_h, const int out_w, 
        const float* inv_trans, cudaStream_t stream) {

    dim3 block(16, 16);
    int grid_x = (out_w + block.x - 1) / block.x;
    int grid_y = (out_h*batch_num + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);


    warp_affine_kernel<T1, T2><<<grid, block, 0, stream>>>(batch_num,
            src, channel, in_h, in_w, dst, out_h, out_w, 
            inv_trans);

}


template void cuda_warp_affine(const int, float*, const int, const int, const int,\
        float*, const int, const int, const float*,  cudaStream_t);
template void cuda_warp_affine(const int, uint8_t*, const int, const int, const int,\
        uint8_t*, const int, const int, const float*, cudaStream_t);
template void cuda_warp_affine(const int, uint8_t*, const int, const int, const int, \
        float*, const int, const int, const float*, cudaStream_t);
