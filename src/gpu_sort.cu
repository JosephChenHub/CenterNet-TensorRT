#include "gpu_common.cuh"
#include "custom.hpp"
#include <vector>
#include <cassert>
#include <iostream>

using namespace std;

#define B_BLOCK_VAR_NUMS 256
#define B_ELEM_PT 4 // each thread read 4 var. first

#define M_BLOCK_VAR_NUMS 256
#define M_ELEM_PT 1

const int B_THREADS_PER_BLOCK = B_BLOCK_VAR_NUMS / B_ELEM_PT;

/// bitonic-sort
template <typename T>
__forceinline__ __device__ void compAndSwap(T* data, const size_t i , const size_t j, const bool dir) {
    if (dir == (data[i] > data[j])) {
        T tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

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




#define B2G(x, i) {compAndSwap(x, i, i+1, ascending);}
#define B4G(x, i) { for(int j = 0; j < 2; ++j) { compAndSwap(x, i+j, j+i+2, ascending); } \
    B2G(x, i)  B2G(x, i+2) }

#define B8G(x, i) { for(int j = 0; j < 4; ++j) { compAndSwap(x, i+j, i+j+4, ascending); } \
    B4G(x, i)  B4G(x, i+4) }

#define B16G(x, i) { for(int j = 0; j < 8; ++j) { compAndSwap(x, i+j, i+j+8, ascending); } \
    B8G(x, i) B8G(x, i+8) }

#define B32G(x, i) { for(int j = 0; j < 16; ++j) { compAndSwap(x, i+j, i+j+16, ascending); } \
    B16G(x, i) B16G(x, i+16) } 

#define B64G(x, i) { for(int j = 0; j < 32; ++j)  { compAndSwap(x, i+j, i+j+32, ascending);} \
    B32G(x, i) B32G(x, i+32) }

#define B128G(x, i) { for(int j = 0; j < 64; ++j) compAndSwap(x, i+j, i+j+64, ascending);\
    B64G(x, i) B64G(x, i+64) }

#define B256G(x, i) { for(int j = 0; j < 128; ++j) compAndSwap(x, i+j, i+j+128, ascending);\
    B128G(x, i) B128G(x, i+128) }

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
__global__ void bitonicBlockSort(T* data, const size_t num, const bool sort_asceding) {
    const size_t tid = threadIdx.x;
    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ T smem[B_BLOCK_VAR_NUMS]; // each block process 256 var.
   
    for(int i = 0; i < B_ELEM_PT; ++i) {
        if (gid*B_ELEM_PT+i < num) smem[tid*B_ELEM_PT + i] = data[gid*B_ELEM_PT + i];
        else {
            if (sort_asceding) set_data(smem[tid*B_ELEM_PT+i], INT_MAX*1.0);
            else set_data(smem[tid*B_ELEM_PT+i], INT_MIN*1.0);
        }
    }
    __syncthreads();
    ///volatile 
    T* sdata = smem + tid * B_ELEM_PT;
    /// 4-group
    bool ascending = sort_asceding;
    B2G(sdata, 0)
    ascending ^= 1;
    B2G(sdata, 2)
    ascending ^= 1;
    __syncthreads();
    if (tid % 2 == 0) { //8-group
        B4G(sdata, 0)
        ascending ^= 1;
        B4G(sdata, 4)
        ascending ^= 1;
    }
    __syncthreads();
    if (tid % 4 == 0) { // 16-group
        B8G(sdata, 0)
        ascending ^= 1;
        B8G(sdata, 8)
        ascending ^= 1;
    }
    __syncthreads();
    if (tid % 8 == 0) { // 32-group
        B16G(sdata, 0)
        ascending ^= 1;
        B16G(sdata, 16)
        ascending ^= 1;
    }
    __syncthreads();
    if (tid % 16 == 0) { //64-group
        B32G(sdata, 0)
        ascending ^= 1;
        B32G(sdata, 32)
        ascending ^= 1;
    }
    __syncthreads();
    if (tid % 32 == 0) { //128-group
        B64G(sdata, 0)
        ascending ^= 1;
        B64G(sdata, 64)
        ascending ^= 1;
    }
    __syncthreads();
    if (tid % 64 == 0) { // 256-group
        B128G(sdata, 0)
        ascending ^= 1;
        B128G(sdata, 128)
        ascending ^= 1;
    }
    __syncthreads();
    /// merge
    if (tid == 0) B256G(sdata, 0)
    __syncthreads();
    /// write to global mem.
    for(int i = 0; i < B_ELEM_PT; ++i) {
        data[gid*B_ELEM_PT + i] = sdata[i];
    }
}

template <typename T>
__global__ void bitonicBlockSortIndices(T* data, size_t* indices, const size_t num, const bool sort_asceding) {
    const size_t tid = threadIdx.x;
    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ T smem[B_BLOCK_VAR_NUMS]; // each block process B_BLOCK_VAR_NUMS var.
    __shared__ size_t sind[B_BLOCK_VAR_NUMS];
   
    for(int i = 0; i < B_ELEM_PT; ++i) {
        if (gid*B_ELEM_PT+i < num) {
            smem[tid*B_ELEM_PT + i] = data[gid*B_ELEM_PT + i];
            sind[tid*B_ELEM_PT + i] = gid*B_ELEM_PT + i;
        }
        else {
            if (sort_asceding) set_data(smem[tid*B_ELEM_PT+i], INT_MAX*1.0);
            else set_data(smem[tid*B_ELEM_PT+i], INT_MIN*1.0);
            sind[tid*B_ELEM_PT+i] = 0;
        }
    }
    __syncthreads();
    ///volatile 
    T* sdata = smem + tid * B_ELEM_PT;
    size_t* sidx  = sind + tid * B_ELEM_PT; 
    /// 4-group
    bool ascending = sort_asceding;
    B2GI(sdata, sidx, 0)
    ascending ^= 1;
    B2GI(sdata, sidx, 2)
    ascending ^= 1;
    __syncthreads();
    if (tid % 2 == 0) { //8-group
        B4GI(sdata, sidx, 0)
        ascending ^= 1;
        B4GI(sdata, sidx, 4)
        ascending ^= 1;
    }
    __syncthreads();
    if (tid % 4 == 0) { // 16-group
        B8GI(sdata, sidx, 0)
        ascending ^= 1;
        B8GI(sdata, sidx, 8)
        ascending ^= 1;
    }
    __syncthreads();
    if (tid % 8 == 0) { // 32-group
        B16GI(sdata, sidx, 0)
        ascending ^= 1;
        B16GI(sdata, sidx, 16)
        ascending ^= 1;
    }
    __syncthreads();
    if (tid % 16 == 0) { //64-group
        B32GI(sdata, sidx, 0)
        ascending ^= 1;
        B32GI(sdata, sidx, 32)
        ascending ^= 1;
    }
    __syncthreads();
    if (tid % 32 == 0) { //128-group
        B64GI(sdata, sidx, 0)
        ascending ^= 1;
        B64GI(sdata,  sidx, 64)
        ascending ^= 1;
    }
    __syncthreads();
    if (tid % 64 == 0) { // 256-group
        B128GI(sdata, sidx, 0)
        ascending ^= 1;
        B128GI(sdata, sidx, 128)
        ascending ^= 1;
    }
    __syncthreads();
    /// merge
    if (tid == 0) B256GI(sdata, sidx, 0)
    __syncthreads();

    for(int i = 0; i < B_ELEM_PT; ++i) {
        data[gid*B_ELEM_PT + i] = sdata[i];
        indices[gid*B_ELEM_PT+i] = sidx[i];
    }
}

template <typename T>
__global__ void bitonicBatchBlockSortIndices(T* data, size_t* indices, const int batch_num,
        const size_t slice_len, const size_t padding_len, 
        const bool sort_asceding) {
    const size_t tid = threadIdx.x;
    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t g_addr, t_addr;
    size_t index;
    __shared__ T smem[B_BLOCK_VAR_NUMS]; // each block process 256 var.
    __shared__ size_t sind[B_BLOCK_VAR_NUMS];

    const size_t batch_id = gid * B_ELEM_PT / padding_len;
    const size_t shift = batch_id * padding_len;

    for(int i = 0; i < B_ELEM_PT; ++i) {
        g_addr = gid * B_ELEM_PT + i;
        t_addr = tid * B_ELEM_PT + i;
        index  = g_addr - shift;
        if (index < slice_len ) {
            smem[t_addr] = data[g_addr];
            sind[t_addr] = index;
        }
        else {
            if (sort_asceding) set_data(smem[t_addr], INT_MAX*1.0);
            else set_data(smem[t_addr], INT_MIN*1.0);
            sind[t_addr] = 0;
        }
    }
    __syncthreads();
    //volatile 
    T* sdata = smem + tid * B_ELEM_PT;
    //volatile 
    size_t* sidx  = sind + tid * B_ELEM_PT; 
    /// 4-group
    bool ascending = sort_asceding;
    B2GI(sdata, sidx, 0)
    ascending ^= 1;
    B2GI(sdata, sidx, 2)
    ascending ^= 1;
    __syncthreads();
    if (tid % 2 == 0) { //8-group
        B4GI(sdata, sidx, 0)
        ascending ^= 1;
        B4GI(sdata, sidx, 4)
        ascending ^= 1;
    }
    __syncthreads();
    if (tid % 4 == 0) { // 16-group
        B8GI(sdata, sidx, 0)
        ascending ^= 1;
        B8GI(sdata, sidx, 8)
        ascending ^= 1;
    }
    __syncthreads();
    if (tid % 8 == 0) { // 32-group
        B16GI(sdata, sidx, 0)
        ascending ^= 1;
        B16GI(sdata, sidx, 16)
        ascending ^= 1;
    }
    __syncthreads();
    if (tid % 16 == 0) { //64-group
        B32GI(sdata, sidx, 0)
        ascending ^= 1;
        B32GI(sdata, sidx, 32)
        ascending ^= 1;
    }
    __syncthreads();
    if (tid % 32 == 0) { //128-group
        B64GI(sdata, sidx, 0)
        ascending ^= 1;
        B64GI(sdata,  sidx, 64)
        ascending ^= 1;
    }
    __syncthreads();
    if (tid % 64 == 0) { // 256-group
        B128GI(sdata, sidx, 0)
        ascending ^= 1;
        B128GI(sdata, sidx, 128)
        ascending ^= 1;
    }
    __syncthreads();
    /// merge
    if (tid == 0) B256GI(sdata, sidx, 0)
    __syncthreads();

    for(int i = 0; i < B_ELEM_PT; ++i) {
        data[gid*B_ELEM_PT + i] = sdata[i];
        indices[gid*B_ELEM_PT+i] = sidx[i];
    }
}

/// merge-sort
template <typename T>
__host__ __device__ void merge(T* left, const size_t left_len, T* right, const size_t right_len,
        T* dest, bool up) {
    size_t i = 0, j = 0, k = 0;
    while(i < left_len && j < right_len) {
        if(!up) {
            if(left[i] > right[j]) dest[k++] = left[i++];
            else dest[k++] = right[j++];
        } else {
            if(left[i] < right[j]) dest[k++] = left[i++];
            else dest[k++] = right[j++];
        }
    }
    while( i < left_len ) dest[k++] = left[i++];
    while( j < right_len ) dest[k++] = right[j++];
}
 
template <typename T>
__global__ void merge_sort_inplace(T* in, const size_t num, bool up) {

    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t tid = threadIdx.x;
    /// 256 threads per block
    __shared__ T smem[256];
    __shared__ T sout[256]; 

    if (gid < num) smem[tid] = in[gid];
    else {
        if (up) set_data(smem[tid], INT_MAX*1.0);
        else set_data(smem[tid], INT_MIN*1.0);
    }
    __syncthreads();

    if (tid < 128) merge<T>(smem+tid, 1, smem+(tid+128), 1, sout+2*tid, up);
    __syncthreads();
    if (tid < 64) merge<T>(sout+tid*2, 2, sout+(tid+64)*2, 2, smem+4*tid, up);
    __syncthreads();
    if (tid < 32) merge<T>(smem+tid*4, 4, smem+(tid+32)*4, 4, sout+8*tid, up);
    __syncthreads();
    if (tid < 16) merge<T>(sout+tid*8, 8, sout+(tid+16)*8, 8, smem+16*tid, up);
    __syncthreads();
    if (tid < 8) merge<T>(smem+tid*16, 16, smem+(tid+8)*16, 16, sout+32*tid, up);
    __syncthreads();
    if (tid < 4) merge<T>(sout+tid*32, 32, sout+(tid+4)*32, 32, smem+64*tid, up);
    __syncthreads();
    if (tid < 2) merge<T>(smem+tid*64, 64, smem+(tid+2)*64, 64, sout+128*tid, up);
    __syncthreads();
    if (tid < 1) merge<T>(sout+tid*128, 128, sout+(tid+1)*128, 128, smem+256*tid, up);
    __syncthreads();

    in[gid] = smem[tid];
    __syncthreads();
}


template <typename T>
__global__ void merge_blocks_result(T* data, 
        const size_t seg_len, const size_t mid,  
        T* out, const bool up) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(mid == 0) return;
    merge<T>(data + seg_len*tid, seg_len, data+seg_len*(tid+mid), seg_len, out+seg_len*2*tid, up);
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
__global__ void topk_merge_blocks(T* data,  size_t* indices, 
        const int  block_var_num, 
        const int slice_blocks_num, 
        const int padding_blocks_num, 
        const int batch_num,  
        const int K) {
    const size_t tid = threadIdx.x; 
    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t batch_id = gid / padding_blocks_num;
    const size_t valid_id = batch_id * padding_blocks_num + slice_blocks_num;

    size_t r_shift ;
    size_t threads = blockDim.x ;
    /// merge within a block
    while (threads > 1 && tid < (threads >> 1) ) {
        r_shift = gid + (threads>>1) ;
        if (r_shift < valid_id)  { // valid mem. access
            topk_merge_two_blocks<T>(data + block_var_num * gid, indices + block_var_num * gid,
                                 data + block_var_num * r_shift, indices + block_var_num * r_shift, K);
        }
        threads >>= 1;
        __syncthreads();
    }
}


template <typename T>
__global__ void align_mem(T* data, size_t* indices, const int batch_num, 
        const size_t slice_blocks_num, const size_t block_var_num, const int K) {
    if (threadIdx.x == 0) {
        for (int i = 1; i < batch_num; ++i)  {
            for( int j = 0; j < K; ++j ) {
                data[K * i + j] = data[slice_blocks_num*block_var_num*i + j];
                indices[K * i + j] = indices[slice_blocks_num * block_var_num * i + j];
            }
        }
    }
}


template <typename T1, typename T2>
T1 divUp(T1 a, T2 b) {
    return (a + b - 1) / b;
}


template <typename T>
__global__ void merge_seg_kernel(T* in, const size_t num, const int block_var_num,  
        size_t* vol, size_t* shift, const int param_size, T* out, bool up) {
    T v ;
    int k = 0;

    size_t pt [32];
    size_t idx;
    for(int i  = 0; i < param_size; ++i) pt[i] = 0;

    for(size_t m = 0; m < num; ++m) {
        if(up) set_data(v, INT_MAX * 1.0);
        else set_data(v, INT_MIN * 1.0);
        for(int i = 0; i < param_size; ++i) {
            idx = shift[i] * block_var_num + pt[i];
            if (up && pt[i] < vol[i] * block_var_num && idx < num && v > in[idx]) {
                v = in[idx];
                k = i;
            }
            if (!up && pt[i] < vol[i] * block_var_num && idx < num && v < in[idx]) {
                v = in[idx];
                k = i;
            } 
        }
        out[m] = in[shift[k] * block_var_num + pt[k]];
        ++pt[k];
    }
}




template <typename T>
void merge_sorted_arrays(T* in, const size_t num, T* buff, const bool up,
        const int merge_blocks_num, const int block_var_num) {
    /// merge blocks' result
    vector<int> param;
    log2_series(merge_blocks_num, param);

    DoubleBuffer<T> buffers(in, buff);
    size_t seg_len = block_var_num;
    size_t sum = 0;
    size_t shift = 0;
    size_t threads_per_block, num_blocks;
    for(size_t i = 0; i < param.size(); ++i) {
        if (!param[i])  break;
        size_t vol = 2 << (param[i] - 1); 
        sum += vol;
        int threads = vol >> 1; 
        while(threads) {
            threads_per_block = threads > block_var_num ? block_var_num: threads;
            num_blocks = threads > block_var_num ? divUp(threads, threads_per_block) : 1;
            merge_blocks_result<T><<<num_blocks, threads_per_block>>>(buffers.Current(), seg_len, threads, buffers.Alternate(), up);
            buffers.selector = buffers.selector ^ 1;
            threads >>= 1; 
            seg_len <<= 1;
        }
        if (param.size() == 1)  {
            if(buffers.Current() == in) return;
            cudaMemcpy(in, buffers.Current(), sizeof(T)*num, cudaMemcpyDeviceToDevice);
            return; 
        }
        if (buffers.Current() != buff + shift) {
            cudaMemcpy(buff + shift, buffers.Current(), sizeof(T) * vol * block_var_num, cudaMemcpyDeviceToDevice);
        }
        shift += vol * block_var_num;
        buffers = DoubleBuffer<T>(in + shift, buff + shift); 
        seg_len = block_var_num;
    }
    if (!param.back()) {
        cudaMemcpy(buff+shift, in+sum*block_var_num, sizeof(T) * block_var_num, cudaMemcpyDeviceToDevice);
    }
    /// merge multiple sorted arrays in GPU
    size_t h_vol[32];
    size_t h_shift[32];
    for(size_t i = 0; i < param.size(); ++i) {
        h_vol[i] = param[i] > 0 ? 2 << (param[i] - 1) : 1;
        h_shift[i] = (i == 0) ? 0 : h_shift[i-1] + h_vol[i-1]; 
    }

    size_t* d_vol;
    size_t* d_shift;
    cudaMalloc((void**)&d_vol, sizeof(size_t) * param.size());
    cudaMalloc((void**)&d_shift, sizeof(size_t) * param.size());

    cudaMemcpy(d_vol, h_vol, sizeof(size_t) * param.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shift, h_shift, sizeof(size_t) * param.size(), cudaMemcpyHostToDevice);

    merge_seg_kernel<T><<<1, 1>>>(buff, num, block_var_num, d_vol, d_shift, param.size(), in, up);

    cudaFree(d_vol);
    cudaFree(d_shift);
}


template <typename T>
void merge_batch_topk(T* idata, size_t* indices, const int batch_num, const size_t padding_len,  
        const int K, const int slice_blocks_num, const int block_var_num,
        cudaStream_t stream) {
    /// blocks: |0, 1, ..., slice_blocks_num-1 | ... | ... |

    const int threads_per_block = 128;

    int blocks_per_batch = slice_blocks_num;
    int padding_blocks_num = divUp(slice_blocks_num, threads_per_block) * threads_per_block;
    int num_blocks = padding_blocks_num / threads_per_block * batch_num;
    size_t num_per_block = block_var_num;

    /// merge within each block
    while (blocks_per_batch > 1) {
        topk_merge_blocks<T><<<num_blocks, threads_per_block, 0, stream>>>(idata, indices, num_per_block,  \
                blocks_per_batch, padding_blocks_num, batch_num, K);
        num_per_block *= threads_per_block;
        blocks_per_batch = num_blocks/batch_num;
        padding_blocks_num = divUp(blocks_per_batch, threads_per_block) * threads_per_block;
        num_blocks = padding_blocks_num / threads_per_block * batch_num;
    }
}


template <typename T>
void mergeSort(T* in, const size_t num, T* buff,  const bool up) {
    const int threads_per_block = M_BLOCK_VAR_NUMS;
    const int num_blocks = divUp(num, threads_per_block);
    /// sort the sub-arrays
    merge_sort_inplace<T><<<num_blocks, threads_per_block>>>(
            in, num,  up);

    /// merge 
    merge_sorted_arrays<T>(in, num, buff, up, num_blocks, M_BLOCK_VAR_NUMS);
}

template <typename T>
void bitonicSort(T* in, const size_t num, T* buff, const bool ascending) {
    int num_blocks = (num + B_BLOCK_VAR_NUMS - 1) / B_BLOCK_VAR_NUMS;
    /// inplace sort within each block
    bitonicBlockSort<T><<<num_blocks, B_THREADS_PER_BLOCK>>>(in, num, ascending);
    /// merge
    merge_sorted_arrays<T>(in, num, buff, ascending, num_blocks, B_BLOCK_VAR_NUMS);
}




template <typename T>
void bitonicBatchTopK(T* data, size_t* indices, const int batch_num, 
        const size_t slice_len, const int K,
        cudaStream_t stream) {

    ///inplace sort with blocks
    size_t padding_len = (slice_len + B_BLOCK_VAR_NUMS - 1) / B_BLOCK_VAR_NUMS * B_BLOCK_VAR_NUMS;
    int num_blocks = padding_len / B_BLOCK_VAR_NUMS * batch_num;  

    bitonicBatchBlockSortIndices<T><<<num_blocks, B_THREADS_PER_BLOCK, 0, stream>>>(data, indices, batch_num, slice_len, padding_len, false);
    /// merge
    merge_batch_topk<T>(data, indices, batch_num, padding_len, K, num_blocks / batch_num, B_BLOCK_VAR_NUMS, stream);
    if (batch_num > 1) {
        align_mem<T><<<1, 1, 0, stream>>>(data, indices, batch_num,  num_blocks/batch_num, B_BLOCK_VAR_NUMS, K);
    }
}

__global__ void ctdet_decode_kernel(
        float* det, 
        float* scores, size_t* indices, float* wh, float* reg,  
        float* trans,
        const int batch_num, const size_t slice_blocks_num, 
        const size_t block_var_num, const int K, 
        const int height, const int width, 
        const bool reg_exist, 
        const int num_joints, const float thresh
        ) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= batch_num * K) return;
    if (scores[tid] < thresh) return;  //no nms, directly using threshold

    const size_t area  = height * width;
    const size_t batch_id = tid / K;
    const size_t local_id = tid % K;

    const size_t iid = slice_blocks_num * block_var_num * batch_id + local_id;
    const int batch_len = 1 + 6 * K;  
    
    size_t class_id = indices[iid] / area;
    indices[iid] %= area;

    atomicAdd(&det[batch_id * batch_len], 1.0);

    float bias = 0.;
    float xs, ys;
    if (!reg_exist) bias = 0.5;
    
    ys = static_cast<size_t>(indices[iid] / width) * 1.0 + bias;
    xs = static_cast<size_t>(indices[iid] % width) * 1.0 + bias;


    if (reg_exist) { // reg: Nx2xHxW -> Nx2xK
        xs += reg[batch_id*2*area + indices[iid]];
        ys += reg[batch_id*2*area + area + indices[iid]];
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
    det[batch_id * batch_len + local_id * 6 + 1 + 1] = scores[tid];
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
    if (scores[tid] < thresh) return;  //! no nms, directly using threshold

    const int res_num = 2 + 4 + num_joints * 2;
    const int batch_len = (1 + res_num * K);

    const size_t area  = height * width;
    const size_t batch_id = tid / K;
    const size_t local_id = tid % K;
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
    det[batch_id * batch_len + local_id * res_num + 1 + 1] = scores[tid];
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

    bitonicBatchBlockSortIndices<float><<<num_blocks, B_THREADS_PER_BLOCK, 0, stream>>>(heat, indices, batch_num, slice_len, padding_len, false);
    CHECK_LAST_ERR("ctdet_bitonic_sort_kernel");
    /// merge
    merge_batch_topk<float>(heat, indices, batch_num, padding_len, K, num_blocks / batch_num, B_BLOCK_VAR_NUMS, stream);
    CHECK_LAST_ERR("ctdet_merge_batch_topk_kernel");
    ///
    ctdet_decode_kernel<<<divUp(K * batch_num, 128), 128, 0, stream>>>(
            det, heat, indices, 
            wh, reg, inv_trans,    
            batch_num, num_blocks/batch_num, 
            B_BLOCK_VAR_NUMS, K, height, width, reg_exist,
            num_classes, threshold);

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
    
    /// get the Top-K of the heat map
    bitonicBatchBlockSortIndices<float><<<heat_num_blocks, B_THREADS_PER_BLOCK, 0, stream>>>(heat, heat_ind, batch_num, heat_slice_len, heat_padding_len, false);
    CHECK_LAST_ERR("heat_bitonic_sort_kernel");

    merge_batch_topk<float>(heat, heat_ind, batch_num, heat_padding_len, K, heat_num_blocks / batch_num, B_BLOCK_VAR_NUMS, stream);
    CHECK_LAST_ERR("heat_merge_batch_topk_kernel");

    /// get the channel Top-K of hm_hp
    if (hm_hp_exist) {
        bitonicBatchBlockSortIndices<float><<<hm_num_blocks, B_THREADS_PER_BLOCK, 0, stream>>>(hm_hp, hm_ind, hm_batch_num, \
                hm_slice_len, hm_padding_len, false);
        CHECK_LAST_ERR("hm_bitonic_sort_kernel");
        merge_batch_topk<float>(hm_hp, hm_ind, hm_batch_num, hm_padding_len, K, hm_num_blocks / hm_batch_num, B_BLOCK_VAR_NUMS, stream);
        CHECK_LAST_ERR("heat_merge_batch_topk_kernel");
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

template void mergeSort<int>(int* , const size_t , int*,  const bool);
template void mergeSort<float>(float*, const size_t, float*,  const bool);
template void mergeSort<double>(double*, const size_t, double*,  const bool);
template void mergeSort<Pair<float, size_t>>(Pair<float, size_t>*, const size_t, Pair<float, size_t>*, const bool);


template void bitonicSort<int>(int *, const size_t, int*, const bool);
template void bitonicSort<float>(float *, const size_t, float*, const bool);
template void bitonicSort<double>(double *, const size_t, double*, const bool);
template void bitonicSort<Pair<float, size_t>>(Pair<float, size_t>*, const size_t, Pair<float, size_t>*, const bool);


template void bitonicBatchTopK<int>(int*, size_t*, const int, const size_t, const int, cudaStream_t);
template void bitonicBatchTopK<float>(float*, size_t*, const int, const size_t, const int, cudaStream_t);
template void bitonicBatchTopK<double>(double*, size_t*, const int, const size_t, const int, cudaStream_t);

