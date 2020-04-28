/*
 * modified from https://github.com/anilshanbhag/gpu-topk
 */ 

#include "sharedmem.hpp"
#include <algorithm>
#include "topk_gpu.hpp"
#include <vector>
#include <cassert>
#include "gpu_common.cuh"

#include "custom.hpp"


using namespace std;

#define NUM_ELEM_PT 16
#define NUM_ELEM_BITSHIFT 4


#define ORDERV(x,a,b) { bool swap = reverse ^ (x[a]<x[b]); \
    T auxa = x[a]; \
    if (swap) { x[a] = x[b]; x[b] = auxa;  } }


#define B2V(x,a) { ORDERV(x,a,a+1)  }
#define B4V(x,a) { for (int i4=0;i4<2;i4++) { ORDERV(x,a+i4,a+i4+2)  } B2V(x,a) B2V(x,a+2)  }
#define B8V(x,a) { for (int i8=0;i8<4;i8++) { ORDERV(x,a+i8,a+i8+4)  } B4V(x,a) B4V(x,a+4)  }
#define B16V(x,a) { for (int i16=0;i16<8;i16++) { ORDERV(x,a+i16,a+i16+8)  } B8V(x,a) B8V(x,a+8)  }
#define B32V(x,a) { for (int i32=0;i32<16;i32++) { ORDERV(x,a+i32,a+i32+16)  } B16V(x,a) B16V(x,a+16)  }
#define B64V(x,a) { for (int i64=0;i64<32;i64++) { ORDERV(x,a+i64,a+i64+32)  } B32V(x,a) B32V(x,a+32)  }

template<typename T>
__forceinline__
__device__  T get(T* sdata, int i) {
    return sdata[i + (i>>5)];

}


template <typename T>
__forceinline__ __device__ T max(T& a, T& b) {
    return a > b ? a : b;
}


#define set(a,b,c) { int tempIndex = b; a[tempIndex + (tempIndex >> 5)] = c;   }

#define NUM_GROUPS (NUM_ELEM_PT/2)
#define NUM_GROUPS_BITSHIFT (NUM_ELEM_BITSHIFT-1)

#define RUN_64(X) { \
    inc >>= 5; \
    low = t & (inc - 1); \
    tCur = ((t - low) << 6) + low; \
    reverse = ((dir & tCur) == 0); \
    for (int j=0; j<NUM_GROUPS/(32 * X); j++) { \
        for (int i=0; i<64; i++) x[i] = get(sdata, tCur+i*inc); \
        B64V(x,0); \
        for (int i=0; i<64; i++) set(sdata, tCur+i*inc, x[i]); \
    } \
    inc >>= 1; \
}

#define RUN_32(X) { \
    inc >>= 4; \
    low = t & (inc - 1); \
    tCur = ((t - low) << 5) + low; \
    reverse = ((dir & tCur) == 0); \
    for (int j=0; j<NUM_GROUPS/(16 * X); j++) { \
        for (int i=0; i<32; i++) x[i] = get(sdata, tCur+i*inc); \
        B32V(x,0); \
        for (int i=0; i<32; i++) set(sdata, tCur+i*inc, x[i]); \
    } \
    inc >>= 1; \
}

#define RUN_16(X) { \
    inc >>= 3; \
    low = t & (inc - 1); \
    tCur = ((t - low) << 4) + low; \
    reverse = ((dir & tCur) == 0); \
    for (int j=0; j<NUM_GROUPS/(8 * X); j++) { \
        for (int i=0; i<16; i++) x[i] = get(sdata, tCur+i*inc); \
        B16V(x,0); \
        for (int i=0; i<16; i++) set(sdata, tCur+i*inc, x[i]); \
    } \
    inc >>= 1; \
}

#define RUN_8(X) { \
    inc >>= 2; \
    low = t & (inc - 1); \
    tCur = ((t - low) << 3) + low; \
    reverse = ((dir & tCur) == 0); \
    for (int j=0; j<NUM_GROUPS/(4 * X); j++) { \
        for (int i=0; i<8; i++) x[i] = get(sdata, tCur+i*inc); \
        B8V(x,0); \
        for (int i=0; i<8; i++) set(sdata, tCur+i*inc, x[i]); \
    } \
    inc >>= 1; \
}

#define RUN_4(X) { \
    inc >>= 1; \
    low = t & (inc - 1); \
    tCur = ((t - low) << 2) + low; \
    reverse = ((dir & tCur) == 0); \
    for (int j=0; j<NUM_GROUPS/(2 * X); j++) { \
        for (int i=0;i<4;i++) x[i] = get(sdata, 4*wg*j + tCur + i*inc); \
        B4V(x,0); \
        for (int i=0;i<4;i++) set(sdata, 4*wg*j + tCur + i*inc, x[i]); \
    } \
    inc >>= 1; \
}

#define RUN_2(X) { \
    low = t & (inc - 1); \
    tCur = ((t - low) << 1) + low; \
    reverse = ((dir & tCur) == 0); \
    for (int j=0; j<NUM_GROUPS/(X); j++) { \
        for (int i=0;i<2;i++) x[i] = get(sdata, 2*wg*j + tCur + i*inc); \
        B2V(x,0); \
        for (int i=0;i<2;i++) set(sdata, 2*wg*j + tCur + i*inc, x[i]); \
    } \
    inc >>= 1; \
}

#define REDUCE(X) { \
    tCur = ((t >> klog2) << (klog2 + 1)) + (t & (k-1)); \
    for(int j=0; j<NUM_GROUPS/(X); j++) { \
        x[j] = max(get(sdata, 2*wg*j + tCur), get(sdata, 2*wg*j + tCur + k)); \
    } \
    __syncthreads(); \
    for(int j=0; j<NUM_GROUPS/(X); j++) { \
        set(sdata, wg*j + t, x[j]); \
    } \
}

template<typename T>
__global__ void Bitonic_TopKLocalSortInPlace(T* __restrict__ in, T* __restrict__ out,
        const int k, const int klog2) {
    /*  const int k = K;*/
    /*const int klog2 = KLog2;*/

    // Shared mem size is determined by the host app at run time.
    // For n elements, we have n * 33/32 shared memory.
    // We use this to break bank conflicts.
    SharedMemory<T> smem;
    T* sdata = smem.getPointer();

    const int t = threadIdx.x; // index in workgroup
    const int wg = blockDim.x; // workgroup size = block size, power of 2
    const int gid = blockIdx.x;


    int length = min(NUM_GROUPS, k >> 1);
    int inc = length;
    inc >>= NUM_GROUPS_BITSHIFT;
    int low = t & (inc - 1);
    int dir = length << 1;
    bool reverse;

    T x[NUM_ELEM_PT];

    // Move IN, OUT to block start
    in += NUM_ELEM_PT * gid * wg;

    int tCur = t << NUM_ELEM_BITSHIFT;
    for (int i=0; i<NUM_ELEM_PT; i++) x[i] = in[tCur + i];

    for (int i=0; i<NUM_ELEM_PT; i+=2) {
        reverse = ((i >> 1) + 1)&1;
        B2V(x,i);

    }
    if (k > 2) {
#if NUM_ELEM_PT > 4
        for (int i=0; i<NUM_ELEM_PT; i+=4) {
            reverse = ((i >> 2) + 1)&1;
            B4V(x,i);

        }
        if (k > 4) {
#if NUM_ELEM_PT > 8
            for (int i=0; i<NUM_ELEM_PT; i+=8) {
                reverse = ((i >> 3) + 1)&1;
                B8V(x,i);

            }
            if (k > 8) {
#if NUM_ELEM_PT > 16
                for (int i=0; i<NUM_ELEM_PT; i+=16) {
                    reverse = ((i >> 4) + 1)&1;
                    B16V(x,i);

                }
                if (k > 16) {
#if NUM_ELEM_PT > 32
                    for (int i=0; i<NUM_ELEM_PT; i+=32) {
                        reverse = ((i >> 5) + 1)&1;
                        B32V(x,i);

                    }
                    if (k > 32) {
                        reverse = ((dir & tCur) == 0); B64V(x,0);

                    }
#else
                    reverse = ((dir & tCur) == 0); B32V(x,0);
#endif

                }
#else
                reverse = ((dir & tCur) == 0); B16V(x,0);
#endif

            }
#else
            reverse = ((dir & tCur) == 0); B8V(x,0);
#endif

        }
#else
        reverse = ((dir & tCur) == 0); B4V(x,0);
#endif

    }

    for (int i=0; i<NUM_ELEM_PT; i++) set(sdata, tCur+i, x[i]);

    __syncthreads();

    // Complete the remaining steps to create sorted sequences of length k.
    int mod;
    unsigned int mask;

    for (length=NUM_ELEM_PT; length<k; length<<=1)
    {
        dir = length << 1;
        // Loop on comparison distance (between keys)
        inc = length;
        mod = inc;
        mask = ~(NUM_ELEM_PT/(1) - 1);
        while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 0);

        if (mod & 1)
        {
            RUN_2(1)
                __syncthreads();

        }
        if (mod & 2)
        {
            RUN_4(1)
                __syncthreads();

        }
#if NUM_ELEM_PT > 8
        if (mod & 4)
        {
            RUN_8(1)
                __syncthreads();

        }
#if NUM_ELEM_PT > 16
        if (mod & 8)
        {
            RUN_16(1)
                __syncthreads();

        }
        while (inc > 8)
        {
            RUN_32(1)
                __syncthreads();

        }
#else
        while (inc > 4)
        {
            RUN_16(1)
                __syncthreads();

        }
#endif // NUM_ELEM_PT > 16
#else
        while (inc > 2)
        {
            RUN_8(1)
                __syncthreads();

        }
#endif // NUM_ELEM_PT > 8

    }

    // Step 2: Reduce the size by factor 2 by pairwise comparing adjacent sequences.
    REDUCE(1)
        __syncthreads();
    // End of Step 2;

    // Step 3: Construct sorted sequence of length k from bitonic sequence of length k.
    // We now have n/2 elements.
    length = k >> 1;
    dir = length << 1;
    // Loop on comparison distance (between keys)
    inc = length;
    mod = inc;
    mask = ~(NUM_ELEM_PT/(1) - 1);
    while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 0);

    if (mod & 1)
    {
        RUN_2(2)
            __syncthreads();

    }
#if NUM_ELEM_PT > 4
    if (mod & 2)
    {
        RUN_4(2)
            __syncthreads();

    }
#if NUM_ELEM_PT > 8
    if (mod & 4)
    {
        RUN_8(2)
            __syncthreads();

    }
    while (inc > 4)
    {
        if (t < (wg >> 1)) {
            RUN_16(1)

        } else {
            inc >>= 4;

        }
        __syncthreads();

    }
#else
    while (inc > 2)
    {
        RUN_8(2)
            __syncthreads();

    }
#endif // NUM_ELEM_PT > 16
#else
    while (inc > 1)
    {
        RUN_4(2)
            __syncthreads();

    }
#endif // NUM_ELEM_PT > 8

    // Step 4: Reduce size again by 2.
    REDUCE(2)
        __syncthreads();
    // End of Step 1;

    length = k >> 1;
    dir = length << 1;
    // Loop on comparison distance (between keys)
    inc = length;
    mod = inc;
    mask = ~(NUM_ELEM_PT/(2) - 1);
    while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 1);

#if NUM_ELEM_PT > 4
    if (mod & 1)
    {
        RUN_2(4)
            __syncthreads();

    }
#if NUM_ELEM_PT > 8
    if (mod & 2)
    {
        RUN_4(4)
            __syncthreads();

    }
    while (inc > 2)
    {
        if (t < (wg >> 1)) {
            RUN_8(2)

        } else {
            inc >>= 3;

        }
        __syncthreads();

    }
#else
    while (inc > 1)
    {
        RUN_4(4)
            __syncthreads();

    }
#endif // NUM_ELEM_PT > 16
#else
    while (inc > 0)
    {
        RUN_2(4)
            __syncthreads();

    }
#endif // NUM_ELEM_PT > 8 while (inc > 0)

    // Step 4: Reduce size again by 2.
    REDUCE(4)
        __syncthreads();
    // End of Step 1;

    length = k >> 1;
    dir = length << 1;
    // Loop on comparison distance (between keys)
    inc = length;
    mod = inc;
    mask = ~(NUM_ELEM_PT/(4) - 1);
    while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 2);

    if (mod & 1)
    {
        RUN_2(8)
            __syncthreads();

    }
    while (inc > 0)
    {
        if (t < (wg >> 1)) {
            RUN_4(4)

        } else {
            inc >>= 2;

        }
        __syncthreads();

    }

    out += (NUM_ELEM_PT/16) * gid * wg;
    tCur = ((t >> klog2) << (klog2+1)) + (t&(k-1));
    for (int j=0; j<NUM_GROUPS/8; j++) {
        T x0 = get(sdata, 2*wg*j + tCur);
        T x1 = get(sdata, 2*wg*j + tCur + k);
        out[wg*j + t] = max(x0, x1);

    }

    /*  out += (NUM_ELEM_PT/8) * gid * wg;*/
    //tCur = ((t >> klog2) << (klog2+1)) + (t&(k-1));
    //for (int j=0; j<NUM_GROUPS/4; j++) {
    //T x0 = get(sdata, 2*wg*j + tCur);
    //T x1 = get(sdata, 2*wg*j + tCur + k);
    //out[wg*j + t] = max(x0, x1);
    /*
       }*/

}

    template<typename T>
__global__ void Bitonic_TopKReduce(T* __restrict__ in, T* __restrict__ out,
        const int k, const int klog2)
{
    /*  const int k = K;*/
    /*const int klog2 = KLog2;*/

    // Shared mem size is determined by the host app at run time.
    // For n elements, we have n * 33/32 shared memory.
    // We use this to break bank conflicts.
    SharedMemory<T> smem;
    T* sdata = smem.getPointer();

    const int t = threadIdx.x; // index in workgroup
    const int wg = blockDim.x; // workgroup size = block size, power of 2
    const int gid = blockIdx.x;

    int length = min(NUM_GROUPS, k >> 1);
    int inc = length;
    inc >>= NUM_GROUPS_BITSHIFT;
    int low = t & (inc - 1);
    int dir = length << 1;
    bool reverse;

    T x[NUM_ELEM_PT];

    // Move IN, OUT to block start
    in += NUM_ELEM_PT * gid * wg;

    int tCur = t << NUM_ELEM_BITSHIFT;
    for (int i=0; i<NUM_ELEM_PT; i++) x[i] = in[tCur + i];
    for (int i=0; i<NUM_ELEM_PT; i++) set(sdata, tCur+i, x[i]);

    __syncthreads();

    // Complete the remaining steps to create sorted sequences of length k.
    int mod;
    unsigned int mask;

    length = (k >> 1);
    dir = length << 1;
    // Loop on comparison distance (between keys)
    inc = length;
    mod = inc;
    mask = ~(NUM_ELEM_PT/(1) - 1);
    while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 0);

    if (mod & 1)
    {
        RUN_2(1)
            __syncthreads();

    }
    if (mod & 2)
    {
        RUN_4(1)
            __syncthreads();

    }
#if NUM_ELEM_PT > 8
    if (mod & 4)
    {
        RUN_8(1)
            __syncthreads();

    }
#if NUM_ELEM_PT > 16
    if (mod & 8)
    {
        RUN_16(1)
            __syncthreads();

    }
    while (inc > 8)
    {
        RUN_32(1)
            __syncthreads();

    }
#else
    while (inc > 4)
    {
        RUN_16(1)
            __syncthreads();

    }
#endif // NUM_ELEM_PT > 16
#else
    while (inc > 2)
    {
        RUN_8(1)
            __syncthreads();

    }
#endif // NUM_ELEM_PT > 8

    // Step 2: Reduce the size by factor 2 by pairwise comparing adjacent sequences.
    REDUCE(1)
        __syncthreads();
    // End of Step 2;

    // Step 3: Construct sorted sequence of length k from bitonic sequence of length k.
    // We now have n/2 elements.
    length = k >> 1;
    dir = length << 1;
    // Loop on comparison distance (between keys)
    inc = length;
    mod = inc;
    mask = ~(NUM_ELEM_PT/(1) - 1);
    while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 0);

    if (mod & 1)
    {
        RUN_2(2)
            __syncthreads();

    }
#if NUM_ELEM_PT > 4
    if (mod & 2)
    {
        RUN_4(2)
            __syncthreads();

    }
#if NUM_ELEM_PT > 8
    if (mod & 4)
    {
        RUN_8(2)
            __syncthreads();

    }
    while (inc > 4)
    {
        if (t < (wg >> 1)) {
            RUN_16(1)

        } else {
            inc >>= 4;

        }
        __syncthreads();

    }
#else
    while (inc > 2)
    {
        RUN_8(2)
            __syncthreads();

    }
#endif // NUM_ELEM_PT > 16
#else
    while (inc > 1)
    {
        RUN_4(2)
            __syncthreads();

    }
#endif // NUM_ELEM_PT > 8

    // Step 4: Reduce size again by 2.
    REDUCE(2)
        __syncthreads();
    // End of Step 1;

    length = k >> 1;
    dir = length << 1;
    // Loop on comparison distance (between keys)
    inc = length;
    mod = inc;
    mask = ~(NUM_ELEM_PT/(2) - 1);
    while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 1);

#if NUM_ELEM_PT > 4
    if (mod & 1)
    {
        RUN_2(4)
            __syncthreads();

    }
#if NUM_ELEM_PT > 8
    if (mod & 2)
    {
        RUN_4(4)
            __syncthreads();

    }
    while (inc > 2)
    {
        if (t < (wg >> 1)) {
            RUN_8(2)

        } else {
            inc >>= 3;

        }
        __syncthreads();

    }
#else
    while (inc > 1)
    {
        RUN_4(4)
            __syncthreads();

    }
#endif // NUM_ELEM_PT > 16
#else
    while (inc > 0)
    {
        RUN_2(4)
            __syncthreads();

    }
#endif // NUM_ELEM_PT > 8 while (inc > 0)

    // Step 4: Reduce size again by 2.
    REDUCE(4)
        __syncthreads();
    // End of Step 1;

    length = k >> 1;
    dir = length << 1;
    // Loop on comparison distance (between keys)
    inc = length;
    mod = inc;
    mask = ~(NUM_ELEM_PT/(4) - 1);
    while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 2);

    if (mod & 1)
    {
        RUN_2(8)
            __syncthreads();

    }
    while (inc > 0)
    {
        if (t < (wg >> 1)) {
            RUN_4(4)

        } else {
            inc >>= 2;

        }
        __syncthreads();

    }

    out += (NUM_ELEM_PT/16) * gid * wg;
    tCur = ((t >> klog2) << (klog2+1)) + (t&(k-1));
    for (int j=0; j<NUM_GROUPS/8; j++) {
        T x0 = get(sdata, 2*wg*j + tCur);
        T x1 = get(sdata, 2*wg*j + tCur + k);
        out[wg*j + t] = max(x0, x1);

    }

}

/*
const int tab32[32] = {
    0,  9,  1, 10, 13, 21,  2, 29,
    11, 14, 16, 18, 22, 25,  3, 30,
    8, 12, 20, 28, 15, 17, 24,  7,
    19, 27, 23,  6, 26,  5,  4, 31
};

int log2_32 (uint value) {
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return tab32[(uint)(value*0x07C4ACDD) >> 27];
}

*/


/// d_keys_in & d_keys_buff: sizeof(KeyT) * num_items
/// d_keys_out: sizeof(KeyT) * k
template<typename KeyT>
void bitonicTopK(KeyT *d_keys_in, 
        KeyT* d_keys_buff, 
        const size_t num_items, 
        const size_t k, vector<KeyT>& vec) {
    if (k < 16) {
        printf("Fatal: k must be greater than 16! %s, %d", __FILE__, __LINE__);
        exit(-1);
    }
    DoubleBuffer<KeyT> d_buffer(d_keys_in, d_keys_buff);

    int klog2 = log2_32(k);

    int current = 0;
    int numThreads = num_items;

    int wg_size = 64 > k ? 64 : k;

    numThreads >>= 1; // Each thread processes 2 elements.
    numThreads >>= NUM_GROUPS_BITSHIFT;

    Bitonic_TopKLocalSortInPlace<KeyT><<<numThreads/wg_size, wg_size, ((2*NUM_GROUPS*wg_size*33)/32)*sizeof(KeyT)>>>(d_buffer.Current(), d_buffer.Alternate(), k, klog2);
    current = 1-current;

    // Toggle the buffer index in the double buffer
    d_buffer.selector = d_buffer.selector ^ 1;

    numThreads >>= (1 + NUM_GROUPS_BITSHIFT);

    while (numThreads >= wg_size) {
        Bitonic_TopKReduce<KeyT><<<numThreads/wg_size, wg_size, ((2*NUM_GROUPS*wg_size*33)/32)*sizeof(KeyT)>>>(d_buffer.Current(), d_buffer.Alternate(), k, klog2);

        // Toggle the buffer index in the double buffer
        d_buffer.selector = d_buffer.selector ^ 1;

        numThreads >>= (1 + NUM_GROUPS_BITSHIFT);

    }

    //vector<KeyT> res_vec(2*numThreads*NUM_GROUPS);
    //cudaMemcpy(res_vec.data(), d_buffer.Current(), 2 * numThreads * NUM_GROUPS * sizeof(KeyT), cudaMemcpyDeviceToHost);
    //std::sort(res_vec.begin(), res_vec.end(), std::greater<KeyT>());
    //cudaMemcpy(d_keys_out, res_vec.data(), k * sizeof(KeyT), cudaMemcpyHostToDevice);
    vec.resize(2*numThreads*NUM_GROUPS);
    cudaMemcpy(vec.data(), d_buffer.Current(), 2 * numThreads * NUM_GROUPS * sizeof(KeyT), cudaMemcpyDeviceToHost);
    std::sort(vec.begin(), vec.end(), std::greater<KeyT>());

}


/// merge-sort
template <typename T>
__device__ void merge(T* left, const size_t left_len, T* right, const size_t right_len,
        T* dest, bool greater) {
    size_t i = 0, j = 0, k = 0;
    while(i < left_len && j < right_len) {
        if(greater) {
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
__global__ void merge_sort_kernel(T* in, const size_t num, T* out,  bool greater) {

    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t tid = threadIdx.x;
    /// 256 threads per block
    __shared__ T smem[256];
    __shared__ T sout[256]; 
    smem[tid] = in[gid];
    __syncthreads();

    if (tid < 128) merge<T>(smem+tid, 1, smem+(tid+128), 1, sout+2*tid, greater);
    __syncthreads();
    if (tid < 64) merge<T>(sout+tid*2, 2, sout+(tid+64)*2, 2, smem+4*tid, greater);
    __syncthreads();
    if (tid < 32) merge<T>(smem+tid*4, 4, smem+(tid+32)*4, 4, sout+8*tid, greater);
    __syncthreads();
    if (tid < 16) merge<T>(sout+tid*8, 8, sout+(tid+16)*8, 8, smem+16*tid, greater);
    __syncthreads();
    if (tid < 8) merge<T>(smem+tid*16, 16, smem+(tid+8)*16, 16, sout+32*tid, greater);
    __syncthreads();
    if (tid < 4) merge<T>(sout+tid*32, 32, sout+(tid+4)*32, 32, smem+64*tid, greater);
    __syncthreads();
    if (tid < 2) merge<T>(smem+tid*64, 64, smem+(tid+2)*64, 64, sout+128*tid, greater);
    __syncthreads();
    if (tid < 1) merge<T>(sout+tid*128, 128, sout+(tid+1)*128, 128, smem+256*tid, greater);
    __syncthreads();

    out[gid] = smem[tid];
    __syncthreads();
}


template <typename T>
__global__ void merge_blocks_result(T* data, const size_t num, const size_t seg_len, const size_t mid,  
        T* out, const bool greater) {
    const size_t tid = threadIdx.x;
    merge<T>(data + seg_len*tid, seg_len, data+seg_len*(tid+mid), seg_len, out+seg_len*2*tid, greater);
}




template <typename T1, typename T2>
T1 divUp(T1 a, T2 b) {
    return (a + b - 1) / b;
}

template <typename T>
void mergeSort(T* in, const size_t num, T*& out, T*& out2, bool greater) {
    const size_t threads_per_block = 256;
    const size_t num_blocks = divUp(num, threads_per_block);

    //int k = log2_32(num);
    //size_t power_k = 2 << k;
    ///printf("num:%d, log2:%d, diff:%d\n", num - power_k);


    merge_sort_kernel<T><<<num_blocks, threads_per_block>>>(
            in, num, out, greater);

    /// merge blocks' result
    DoubleBuffer<T> buffers(out, out2);
    int threads = num_blocks >> 1;
    size_t seg_len = threads_per_block;
    buffers.selector = 0;
    while (threads) {
        merge_blocks_result<T><<<1, threads>>>(buffers.Current(), num, seg_len, threads, buffers.Alternate(), greater);
        buffers.selector = buffers.selector ^ 1;
        threads >>= 1; 
        seg_len <<= 1;
    }
    if(buffers.Current() != out2) {
        out2 = buffers.Current();
        out = buffers.Alternate();
    }
}


template <typename T>
void fastTopK(T* data, T* buff, const size_t num, const size_t k, T* out) {
    int log_k = log2_32(k);
    int power_k = 2 << (log_k - 1);
    size_t new_k = k;
    if (power_k != k) { 
        new_k = 2 << log_k;
    }
    assert (new_k < num);

    /// stage 1 : 2 ^ log2(num)
    int log_num = log2_32(num);
    size_t power_num = 2 << (log_num-1);
    size_t diff_num = 0;
    if (power_num != num) {
        diff_num = num - power_num;  
    }
    vector<T> out_buff[4];
    bitonicTopK<T>(data, buff, power_num, new_k, out_buff[0]);
    if (!diff_num) {
        cudaMemcpy(out, out_buff[0].data(), sizeof(T) * k, cudaMemcpyHostToDevice);
        return;
    }
    /// stage 2: diff = num - 2^(log2(num))
    if (diff_num < 1024 || diff_num <= new_k) {
        out_buff[1].resize(power_num + diff_num);
        memcpy(out_buff[1].data(), data+power_num, diff_num * sizeof(T));
        memcpy(out_buff[1].data() + diff_num, out_buff[0].data(), sizeof(T)*new_k);
        std::sort(out_buff[1].begin(), out_buff[1].end(), std::greater<T>());
        cudaMemcpy(out, out_buff[1].data(), sizeof(T) * k, cudaMemcpyHostToDevice);
        return;
    }
    /// stage 3: diff2 = diff - 2^(log2(diff))
    int log_diff = log2_32(diff_num);
    size_t power_diff = 2 << (log_diff-1);
    bitonicTopK<T>(data + power_num, buff+power_num, power_diff, new_k, out_buff[1]);
    size_t diff2 = diff_num - power_diff;
    int log_diff2 = log2_32(diff2);
    size_t power_diff2 = 2 << (log_diff2-1);
    size_t diff3 = diff2 - power_diff2;
    ///printf("diff:%d, diff2:%d, diff3:%d\n", diff_num, diff2, diff3);
    if (diff2 < 1024 || diff2 <= new_k) {
        out_buff[2].resize(new_k * 2 + diff2);
        if (diff2) cudaMemcpy(out_buff[2].data(), data+power_num+power_diff, sizeof(T) * diff2, cudaMemcpyDeviceToHost);
        memcpy(out_buff[2].data() + diff2, out_buff[0].data(), sizeof(T) * new_k );
        memcpy(out_buff[2].data() + new_k + diff2, out_buff[1].data(), sizeof(T) * new_k);
        std::sort(out_buff[2].begin(), out_buff[2].end(), std::greater<T>());
        cudaMemcpy(out, out_buff[2].data(), sizeof(T) * k, cudaMemcpyHostToDevice);
        return;
    }
    bitonicTopK<T>(data+power_num+power_diff, buff+power_num+power_diff, power_diff2, new_k, out_buff[2]);
    out_buff[3].resize(new_k * 3 + diff3);
    cudaMemcpy(out_buff[3].data(), data+power_num+power_diff+power_diff2, diff3 * sizeof(T), cudaMemcpyDeviceToHost);
    memcpy(out_buff[3].data() + diff3, out_buff[0].data(), sizeof(T) * new_k);
    memcpy(out_buff[3].data() + diff3 + new_k, out_buff[1].data(), sizeof(T) * new_k);
    memcpy(out_buff[3].data() + diff3 + new_k *2, out_buff[2].data(), sizeof(T) * new_k);
    std::sort(out_buff[3].begin(), out_buff[3].end(), std::greater<T>());
    cudaMemcpy(out, out_buff[3].data(), sizeof(T)*k, cudaMemcpyHostToDevice);
}




template void bitonicTopK<float>(float*, float*, const size_t, const size_t, vector<float>&);
template void bitonicTopK<double>(double*, double*, const size_t, const size_t, vector<double>&);
template void bitonicTopK<int>(int*, int*, const size_t, const size_t,  vector<int>&);


template void fastTopK<float>(float* , float* , const size_t , const size_t , float* );
template void fastTopK<double>(double* , double* , const size_t , const size_t , double* );
template void fastTopK<int>(int* , int* , const size_t , const size_t , int* );



