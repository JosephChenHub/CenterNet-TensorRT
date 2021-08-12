#include "gpu_common.cuh"
#include "custom.hpp"
#include <vector>
#include <cassert>
#include <iostream>

#include "decode.hpp"


using namespace std;


#define B_ELEM_PT 16 // each thread process 16 var. firstly
#define B_ELEM_BITSHIFT 4 // log2(B_ELEM_PT)

#define B_THREADS_PER_BLOCK  64
#define B_BLOCK_VAR_NUMS (B_ELEM_PT*B_THREADS_PER_BLOCK)



template <typename T>
__forceinline__ __device__ void compAndSwapIndices(T* data, size_t* indices, 
        const size_t i, const size_t j, const bool dir) {
    if (dir == (data[i] > data[j])) {
        T tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
        size_t idx = indices[i];
        indices[i] = indices[j];
        indices[j] = idx;
    }
}

template <typename T1, typename T2>
__forceinline__ __device__ void set_data(T1& , T2);

template <>
__forceinline__ __device__ void set_data(float& data, double value) {
    data = static_cast<float>(value);
}

template <>
__forceinline__ __device__ void set_data(double& data, double value) {
    data = static_cast<double>(value);
}

template <>
__forceinline__ __device__ void set_data(int& data, double value) {
    data = static_cast<int>(value);
}

template <>
__forceinline__ __device__ void set_data(Pair<float, size_t>& data, double value) {
    data.k = static_cast<float>(value);
    data.v = INT_MAX;
}


#define B2GI(x, k, i) {compAndSwapIndices(x, k, i, i+1, ascending);}
#define B4GI(x, k, i) { for(int j = 0; j < 2; ++j) { compAndSwapIndices(x, k, i+j, j+i+2, ascending); } \
    B2GI(x, k, i)  B2GI(x, k, i+2) }

#define B8GI(x,k, i) { for(int j = 0; j < 4; ++j) { compAndSwapIndices(x, k, i+j, i+j+4, ascending); } \
    B4GI(x, k, i)  B4GI(x, k, i+4) }

#define B16GI(x,k, i) { for(int j = 0; j < 8; ++j) { compAndSwapIndices(x, k, i+j, i+j+8, ascending); } \
    B8GI(x, k, i) B8GI(x, k, i+8) }

#define B32GI(x,k, i) { for(int j = 0; j < 16; ++j) { compAndSwapIndices(x, k, i+j, i+j+16, ascending); } \
    B16GI(x, k, i) B16GI(x, k, i+16) } 

#define B64GI(x, k, i) { for(int j = 0; j < 32; ++j)  { compAndSwapIndices(x, k, i+j, i+j+32, ascending);} \
    B32GI(x, k, i) B32GI(x, k, i+32) }

#define B128GI(x, k, i) { for(int j = 0; j < 64; ++j) compAndSwapIndices(x, k, i+j, i+j+64, ascending);\
    B64GI(x, k, i) B64GI(x, k, i+64) }

#define B256GI(x, k, i) { for(int j = 0; j < 128; ++j) compAndSwapIndices(x, k, i+j, i+j+128, ascending);\
    B128GI(x, k, i) B128GI(x, k, i+128) }




template <typename T>
__global__ void bitonicLocalSortIndices(T* data, size_t* indices, const int batch_num,
        const size_t slice_len, const size_t padding_len, 
        const int K) {
    const int tid = threadIdx.x;
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ T smem[B_BLOCK_VAR_NUMS]; 
    __shared__ size_t sind[B_BLOCK_VAR_NUMS];

    int t_addr = tid << B_ELEM_BITSHIFT;
    int g_addr = gid << B_ELEM_BITSHIFT; 

    bool ascending;

    // we firt read the global mem. to the buffer. 
    T rx[B_ELEM_PT];
    size_t ri[B_ELEM_PT];
    for(int i = 0; i < B_ELEM_PT; ++i) {
        size_t index = (g_addr+i) % padding_len;
        if (index < slice_len ) {
            rx[i] = data[g_addr + i];
            ri[i] = index;
        }
        else {
            rx[i] = INT_MIN*1.0;
            ri[i] = INT_MAX;
        }
    }

    ascending = false;
    for(int i = 0; i < B_ELEM_PT; i += 2) {
	B2GI(rx, ri, i); 
	ascending ^= 1;
    }
    for(int i = 0; i < B_ELEM_PT; i += 4) {
	B4GI(rx, ri, i);
	ascending ^= 1;
    }
    for(int i = 0; i < B_ELEM_PT; i += 8) {
	B8GI(rx, ri, i);
	ascending ^= 1;
    }

    // write to the shared memory 
    for(int i = 0; i < B_ELEM_PT; ++i) {
	smem[t_addr + i] = rx[i];
	sind[t_addr + i] = ri[i];
    }
    __syncthreads();

   T* sdata = smem + t_addr;
   size_t * sidx = sind + t_addr;
   ascending = tid & 1;
   B16GI(sdata, sidx, 0);
   __syncthreads();

   if (tid % 2 == 0 ) {
      ascending = (tid >> 1) & 1;
      B32GI(sdata, sidx, 0); 
   }
   __syncthreads();
   if (tid % 4 == 0 ) {
      ascending = (tid >> 2) & 1;
      B64GI(sdata, sidx, 0);
   }
   __syncthreads();
   if (tid % 8 == 0 ) {
      ascending = (tid >> 3) & 1;
      B128GI(sdata, sidx, 0); 
   }
   __syncthreads();


    /// merge  down-up-down-up-down-up-down-up
    ///        down-up-down-up
    ///        down-up
    ///        down

    int seg_num = blockDim.x * B_ELEM_PT / K;
    size_t lo, hi;
    for( ; seg_num > 1; )  {
        if (tid < seg_num>> 1) {
            lo = K * tid; 
            hi = (seg_num - 1 - tid) * K;

            for(int j = 0; j < K; ++j) compAndSwapIndices(smem, sind, lo + j, hi + j, false);
            ascending = tid & 1;
            B128GI(smem, sind, lo);
         } 
        seg_num >>= 1;
        __syncthreads();
    }


    for(int i = 0; i < B_ELEM_PT; ++i) {
        data[gid*B_ELEM_PT + i] = sdata[i];
        indices[gid*B_ELEM_PT+i] = sidx[i];
    }

}





template <typename T>
__device__ void topk_merge_two_blocks(T* left, size_t* i_left,
        T* right, size_t* i_right, const int K) {
    int i, j;
    T tmp;
    size_t i_tmp;

    if (left[K-1] > right[0]) return;
    if (left[0] < right[K-1]) {
        for(i = 0; i < K; ++i) { left[i] = right[i]; i_left[i] = i_right[i]; }
        return;
    }
    for( i = 0; i < K; ++i ) {
        if (left[i] > right[0]) continue;
        tmp = left[i];
        i_tmp = i_left[i];
        left[i] = right[0];
        i_left[i] = i_right[0];
        for(j = 1; j < K; ++j) {
            if (tmp < right[j]) {
                right[j-1] = right[j];
                i_right[j-1] = i_right[j];
            } else {
                right[j-1] = tmp; 
                i_right[j-1] = i_tmp;
                break;
            } 
        }
    }
}



template <typename T>
__global__ void topk_reduce_blocks(T* data,  size_t* indices, 
        const size_t  num_per_block, 
        const int blocks_per_batch,  
        const int padding_blocks_per_batch,  
        const int batch_num,  
        const int K, const int power_k) {
    const size_t tid = threadIdx.x; 
    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x * 2;

    const size_t batch_id = gid / padding_blocks_per_batch;
    const size_t local_id = gid % padding_blocks_per_batch; 
    if(local_id >= blocks_per_batch) return;


    size_t l_shift = batch_id * blocks_per_batch + local_id; 
    size_t r_shift;

    int half, threads = blockDim.x * 2;
    const bool ascending = false;

    /// merge within a block
    while (threads > 1 && tid < (threads >> 1)) {
        half = (threads>>1) ;
        if (tid + half < blocks_per_batch)  { // valid mem. access
            r_shift = l_shift + half;
            /// topk_merge_two_blocks<T>(data + num_per_block * l_shift, indices + num_per_block * l_shift, \
                    data + num_per_block * r_shift, indices + num_per_block * r_shift, K);
            /// like bitonic sort, this is faster than topk_merge_two_blocks
            for(int i = 0; i < power_k; ++i) {
                if(data[num_per_block*l_shift + i] < data[power_k + num_per_block*r_shift - 1 - i]) {
                    data[num_per_block*l_shift+i] = data[power_k + num_per_block*r_shift - 1 -i];
                    indices[num_per_block*l_shift+i] = indices[power_k + num_per_block*r_shift-1-i];
                }
            }
            B128GI(data + num_per_block * l_shift, indices + num_per_block*l_shift, 0)
        }
        threads >>= 1;
        __syncthreads();
    }
}




template <typename T1, typename T2>
__inline__ T1 divUp(T1 a, T2 b) {
    return (a + b - 1) / b;
}



__inline__ int get_threads(int num) {
    if (num > 512) return 256;
    if (num > 256) return 128;
    if (num > 128) return 64;
    return 32;
}



template <typename T>
void merge_batch_topk(T* idata, size_t* indices, const int batch_num, const size_t padding_len,  
        const int K, const int slice_blocks_num, const int block_var_num,
        cudaStream_t stream) {

    assert(K <= 128);

    size_t blocks_per_batch = slice_blocks_num;
    //int threads_per_block = slice_blocks_num > 128 ? 128 : 64; 
    int threads_per_block = get_threads(slice_blocks_num); 

    size_t padding_blocks_per_batch = divUp(blocks_per_batch, threads_per_block*2) * threads_per_block * 2;
    size_t num_blocks = padding_blocks_per_batch / threads_per_block / 2 * batch_num;
    size_t num_per_block = block_var_num;

    int log_k = log2_32(K);
    int power_k = 2 << (log_k -1);
    if (power_k != K) {
        power_k = 2 << log_k;
    }

    /// merge within each block
    while (blocks_per_batch > 1) {
        topk_reduce_blocks<T><<<num_blocks, threads_per_block, 0, stream>>>(idata, indices, num_per_block,  \
                blocks_per_batch, padding_blocks_per_batch, batch_num, K, power_k);
        num_per_block *= threads_per_block;
        num_per_block <<= 1;
        // threads_per_block = blocks_per_batch > 128 ? 128 : 64;
        threads_per_block = get_threads(blocks_per_batch); 
        blocks_per_batch = num_blocks / batch_num;
        padding_blocks_per_batch = divUp(blocks_per_batch, threads_per_block*2) * threads_per_block;
        padding_blocks_per_batch <<= 1;
        num_blocks = padding_blocks_per_batch / threads_per_block * batch_num;
        num_blocks >>= 1;
    }
}






template <typename T>
void bitonicBatchTopK(T* data, size_t* indices, const int batch_num, 
        const size_t slice_len, const int K,
        cudaStream_t stream) {

    size_t padding_len = (slice_len + B_BLOCK_VAR_NUMS - 1) / B_BLOCK_VAR_NUMS * B_BLOCK_VAR_NUMS;
    int num_blocks = padding_len / B_BLOCK_VAR_NUMS * batch_num;  

    int log_k = log2_32(K);
    int power_k = 2 << (log_k -1);
    if (power_k != K) {
        power_k = 2 << log_k;
    }

    /// local sort
    bitonicLocalSortIndices<T><<<num_blocks, B_THREADS_PER_BLOCK, 0, stream>>>(data, indices, batch_num, 
            slice_len, padding_len, power_k);

    /// merge
    merge_batch_topk<T>(data, indices, batch_num, padding_len, K, num_blocks / batch_num, B_BLOCK_VAR_NUMS, stream);
}

__global__ void ctdet_decode_kernel(
        float* det, 
        float* scores, size_t* indices, float* wh, float* reg,  
        float* trans,
        const int batch_num, const int num_classes, const size_t slice_blocks_num, 
        const size_t block_var_num, const int K, 
        const int height, const int width, 
        const bool reg_exist,  const float thresh
        ) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= batch_num * K) return;
//    if (scores[tid] < thresh) return;  //no nms, directly using threshold

    const size_t area  = height * width;
    const size_t batch_id = tid / K;
    const size_t local_id = tid % K;

    float score = scores[batch_id * area * num_classes + local_id];
    if (score < thresh) return; // no nms

    const size_t iid = slice_blocks_num * block_var_num * batch_id + local_id;
    const int batch_len = 1 + 6 * K;  

    size_t class_id = indices[iid] / area;
    indices[iid] %= area;

    atomicAdd(&det[batch_id * batch_len], 1.0);

    float xs, ys;
    ys = static_cast<size_t>(indices[iid] / width) * 1.0;
    xs = static_cast<size_t>(indices[iid] % width) * 1.0;

    if (reg_exist) { // reg: Nx2xHxW -> Nx2xK
        xs += reg[batch_id*2*area + indices[iid]];
        ys += reg[batch_id*2*area + area + indices[iid]];
    } else {
        xs += 0.5;
        ys += 0.5;
    }
    float wh1 = wh[batch_id*2*area + indices[iid]] / 2.0;
    float wh2 = wh[batch_id*2*area + area + indices[iid]] / 2.0;

    float t0, t1, t2, t3;
    float tt0, tt1, tt2, tt3;
    t0 = xs - wh1;
    t1 = ys - wh2;
    t2 = xs + wh1;
    t3 = ys + wh2;

    /// inverse-warpAffine
    tt0 = trans[0] * t0 + trans[1] * t1 + trans[2]; 
    tt1 = trans[3] * t0 + trans[4] * t1 + trans[5];
    tt2 = trans[0] * t2 + trans[1] * t3 + trans[2];
    tt3 = trans[3] * t2 + trans[4] * t3 + trans[5];

    //printf("id:%d, score:%.4f, cls:%d, box:(%.1f, %.1f, %.1f, %.1f)\n", tid,  scores[tid], class_id, tt0, tt1, tt2, tt3);
    /// det: N* (1 + 6*K)

    det[batch_id * batch_len + local_id * 6 + 0 + 1] = class_id;
    det[batch_id * batch_len + local_id * 6 + 1 + 1] = score;
    det[batch_id * batch_len + local_id * 6 + 2 + 1] = tt0;
    det[batch_id * batch_len + local_id * 6 + 3 + 1] = tt1;
    det[batch_id * batch_len + local_id * 6 + 4 + 1] = tt2;
    det[batch_id * batch_len + local_id * 6 + 5 + 1] = tt3;

}

__global__ void pose_decode_kernel(
        float* det, 
        float* scores, size_t* indices, float* wh, float* reg,  
        float* hps, float* hm_hp, size_t* hm_ind, 
        float* hp_offset, 
        float* trans,
        const int batch_num, const size_t slice_blocks_num, 
        const size_t block_var_num, const int K, 
        const int height, const int width, 
        const bool reg_exist, const bool hm_hp_exist, 
        const int num_joints, const float thresh
        ) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= batch_num * K) return;
//    if (scores[tid] < thresh) return;  //! no nms, directly using threshold

    const int res_num = 2 + 4 + num_joints * 2;
    const int batch_len = (1 + res_num * K);

    const size_t area  = height * width;
    const size_t batch_id = tid / K;
    const size_t local_id = tid % K;
    float score = scores[batch_id * area  + local_id];
    if (score < thresh) return;  // no nms

    atomicAdd(&det[batch_id * batch_len], 1.0); //! number of det

    const size_t iid = slice_blocks_num * block_var_num * batch_id + local_id;

    size_t class_id = indices[iid] / area;
    indices[iid] %= area;

    float xs, ys;
    float p0, p1;
    float t0, t1, t2, t3;
    float tt0, tt1, tt2, tt3;
    float bias[2];
    ys = static_cast<size_t>(indices[iid] / width) * 1.0;
    xs = static_cast<size_t>(indices[iid] % width) * 1.0;

    if (reg_exist) { // reg: Nx2xHxW -> Nx2xK
        bias[0] = reg[batch_id*2*area + indices[iid]];
        bias[1] = reg[batch_id*2*area + area + indices[iid]];
    } else {
        bias[0] = 0.5; bias[1] = 0.5;
    }
    float wh1 = wh[batch_id*2*area + indices[iid]] / 2.0;
    float wh2 = wh[batch_id*2*area + area + indices[iid]] / 2.0;

    t0 = xs - wh1 + bias[0];
    t1 = ys - wh2 + bias[1];
    t2 = xs + wh1 + bias[0];
    t3 = ys + wh2 + bias[1];

    /// inverse-warpAffine
    tt0 = trans[0] * t0 + trans[1] * t1 + trans[2]; 
    tt1 = trans[3] * t0 + trans[4] * t1 + trans[5];
    tt2 = trans[0] * t2 + trans[1] * t3 + trans[2];
    tt3 = trans[3] * t2 + trans[4] * t3 + trans[5];
    det[batch_id * batch_len + local_id * res_num + 0 + 1] = class_id;
    det[batch_id * batch_len + local_id * res_num + 1 + 1] = score; 
    det[batch_id * batch_len + local_id * res_num + 2 + 1] = tt0;
    det[batch_id * batch_len + local_id * res_num + 3 + 1] = tt1;
    det[batch_id * batch_len + local_id * res_num + 4 + 1] = tt2;
    det[batch_id * batch_len + local_id * res_num + 5 + 1] = tt3;

    /// key points
    for(int i = 0; i < num_joints; ++i) {
        p0 = hps[batch_id*num_joints*2*area + i*2*area + indices[iid]] + xs;
        p1 = hps[batch_id*num_joints*2*area + (i*2+1)*area + indices[iid]] + ys;

        /// find the most closed point with a confidence > 0.1 
        if (hm_hp_exist) {
            float min_ds = static_cast<float>(INT_MAX);
            float near_xs = min_ds, near_ys = min_ds;

            // hm_hp: N x 17 x 128 x 128 
            float hm_hp_score, diff = min_ds;
            float hm_xs, hm_ys;
            size_t ind_tmp;
            for(int j = 0; j < K; ++j) {
                hm_hp_score = hm_hp[batch_id * num_joints * area + i * area + j];
                if (hm_hp_score < 0.1)  continue;
                ind_tmp = hm_ind[batch_id*num_joints*area + i * area + j] % area;
                hm_ys = static_cast<size_t>(ind_tmp / width) * 1.0 + hp_offset[batch_id*2*area + area + j];
                hm_xs = static_cast<size_t>(ind_tmp % width) * 1.0 + hp_offset[batch_id*2*area + j];
                diff = fabs(p0 - hm_xs) + fabs(p1 - hm_ys);
                if (diff < min_ds) {
                    min_ds = diff;
                    near_xs = hm_xs;
                    near_ys = hm_ys;
                }
            }
            if (near_xs > t0 && near_xs < t2 && near_ys > t1 &&
                    near_ys < t3 && diff < max(t2-t0, t3-t1) * 0.5) {
                p0 = near_xs;
                p1 = near_ys;
            }
        }

        tt0 = trans[0] * p0 + trans[1] * p1 + trans[2];
        tt1 = trans[3] * p0 + trans[4] * p1 + trans[5];

        det[batch_id * batch_len + local_id * res_num + i*2 + 7] = tt0;
        det[batch_id * batch_len + local_id * res_num + i*2+1 + 7] = tt1;
    }
}

void ctdet_decode(
        float* det,  
        float* wh, float* reg, 
        float* heat, size_t* indices,    
        float* inv_trans, 
        const int batch_num, const int num_classes,  
        const int height, const int width, 
        const int K, const float threshold, 
        const bool reg_exist, const bool cat_spec_wh,
        cudaStream_t stream) {

    ///inplace sort with blocks
    const size_t slice_len = height * width * num_classes;
    const size_t padding_len = (slice_len + B_BLOCK_VAR_NUMS - 1) / B_BLOCK_VAR_NUMS * B_BLOCK_VAR_NUMS;
    int num_blocks = padding_len / B_BLOCK_VAR_NUMS * batch_num;  
    int log_k = log2_32(K);
    int power_k = 2 << (log_k -1);
    if (power_k != K) {
        power_k = 2 << log_k;
    }
    bitonicLocalSortIndices<float><<<num_blocks, B_THREADS_PER_BLOCK, 0, stream>>>(heat, indices, batch_num, slice_len, padding_len, power_k);
    CHECK_LAST_ERR("ctdet_bitonic_batch_topk_kernel");
    /// merge
    merge_batch_topk<float>(heat, indices, batch_num, padding_len, K, num_blocks / batch_num, B_BLOCK_VAR_NUMS, stream);
    CHECK_LAST_ERR("ctdet_merge_batch_topk_kernel");
    ///
    ctdet_decode_kernel<<<divUp(K * batch_num, 128), 128, 0, stream>>>(
            det, heat, indices, 
            wh, reg, inv_trans,    
            batch_num, num_classes, num_blocks/batch_num, 
            B_BLOCK_VAR_NUMS, K, height, width, reg_exist,
            threshold);

    CHECK_LAST_ERR("ctdet_decode_kernel");
}


void multi_pose_decode(
        float* det,  
        float* heat, float* wh, float* reg, 
        float* hps, float* hm_hp, float* hp_offset, 
        size_t* heat_ind, size_t* hm_ind, 
        float* inv_trans, 
        const int batch_num, const int num_joints, 
        const int height, const int width, 
        const int K, const float threshold, 
        const bool reg_exist, const bool hm_hp_exist, 
        cudaStream_t stream) {

    const size_t area = height * width;
    const size_t heat_slice_len = area * 1; 
    const size_t heat_padding_len = (heat_slice_len + B_BLOCK_VAR_NUMS - 1) / B_BLOCK_VAR_NUMS * B_BLOCK_VAR_NUMS;
    const int heat_num_blocks = heat_padding_len / B_BLOCK_VAR_NUMS * batch_num;  

    const size_t hm_slice_len = area;
    const size_t hm_padding_len = (area + B_BLOCK_VAR_NUMS-1)/B_BLOCK_VAR_NUMS * B_BLOCK_VAR_NUMS;
    const int hm_batch_num = batch_num * num_joints;
    const int hm_num_blocks = hm_padding_len / B_BLOCK_VAR_NUMS * batch_num * num_joints;
    int log_k = log2_32(K);
    int power_k = 2 << (log_k -1);
    if (power_k != K) {
        power_k = 2 << log_k;
    }

    /// get the Top-K of the heat map
    bitonicLocalSortIndices<float><<<heat_num_blocks, B_THREADS_PER_BLOCK, 0, stream>>>(heat, heat_ind, batch_num, heat_slice_len, heat_padding_len, power_k);
    CHECK_LAST_ERR("pose_bitonic_topk_kernel");

    merge_batch_topk<float>(heat, heat_ind, batch_num, heat_padding_len, K, heat_num_blocks / batch_num, B_BLOCK_VAR_NUMS, stream);
    CHECK_LAST_ERR("pose_merge_topk_kernel");

    /// get the channel Top-K of hm_hp
    if (hm_hp_exist) {
        bitonicLocalSortIndices<float><<<hm_num_blocks, B_THREADS_PER_BLOCK, 0, stream>>>(hm_hp, hm_ind, hm_batch_num, \
                hm_slice_len, hm_padding_len, power_k);
        CHECK_LAST_ERR("pose_bitonic_topk_kernel");
        merge_batch_topk<float>(hm_hp, hm_ind, hm_batch_num, hm_padding_len, K, hm_num_blocks / hm_batch_num, B_BLOCK_VAR_NUMS, stream);
        CHECK_LAST_ERR("pose_merge_topk_kernel");
    }

    /// decode 
    pose_decode_kernel<<<divUp(K * batch_num, 128), 128, 0, stream>>>(
            det, heat, heat_ind, 
            wh, reg, hps, 
            hm_hp, hm_ind, 
            hp_offset, inv_trans,    
            batch_num, heat_num_blocks/batch_num, 
            B_BLOCK_VAR_NUMS, K, height, width, 
            reg_exist, hm_hp_exist, 
            num_joints, threshold);

    CHECK_LAST_ERR("pose_decode_kernel");
}



template void bitonicBatchTopK<int>(int*, size_t*, const int, const size_t, const int, cudaStream_t);
template void bitonicBatchTopK<float>(float*, size_t*, const int, const size_t, const int, cudaStream_t);
template void bitonicBatchTopK<double>(double*, size_t*, const int, const size_t, const int, cudaStream_t);

