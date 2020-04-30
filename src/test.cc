#include <gtest/gtest.h>
#include <vector>
//#include <fstream>
//#include <sstream>
//#include <cstring>
#include <memory>
#include <math.h>

#include <stdlib.h> // rand
#include <time.h> 

#include <cuda_runtime_api.h> // cuda
#include <opencv2/opencv.hpp>

#include "topk_gpu.hpp"
#include "topk_cpu.hpp"
#include "gpu_sort.hpp"
#include "det_kernels.hpp"
#include "custom.hpp"
#include "gpu_common.cuh"

//#include <ATen/ATen.h> // libtorch, cpu version 

//#include "common/logger.h"

using namespace std;
using namespace cv;
//using namespace nvinfer1;



/*
struct NvInferDeleter {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};
*/

/*
vector<string> split_str(string& str, string pattern) {
    vector<string> res;
    if (pattern.empty()) return res;
    size_t start = 0, index = str.find_first_of(pattern, 0);
    while (index != str.npos) {
        if (start != index) res.push_back(str.substr(start, index - start));
        start = index + 1;
        index = str.find_first_of(pattern, start);
    }
    if (!str.substr(start).empty()) res.push_back(str.substr(start));
    return res;
}
*/


TEST(Foo, test) {
   const int a = 1;
   ASSERT_EQ(1, a);
   cudaSetDevice(0);
}

template <typename T>
void merge_topk(T* left, size_t* i_left,
        T* right, size_t* i_right, const int K) {
    int i, j;
    T tmp;
    size_t i_tmp;

    if (left[K-1] > right[0]) return;
    if (left[0] < right[K-1]) {
        for(i = 0; i < K; ++i) left[i] = right[i];
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

TEST(CPU, merge_topk) {
    typedef float T;

    const int K = 100;
    const int N = 5120;

    vector<T> data(N);
    vector<Pair<T, size_t>> buff(N);
    vector<size_t> i_data(N);
    std::default_random_engine generator;
    std::normal_distribution<T> distribution(1000., 5.);
    for(size_t i = 0; i < N; ++i) {
        data[i] = distribution(generator);
        i_data[i] = i;
        buff[i].k = data[i];
        buff[i].v = i;
    }
    size_t shift = N / 2;
    std::sort(data.begin(), data.begin() + shift, std::greater<T>());
    std::sort(data.begin() + shift, data.end(), std::greater<T>());
    merge_topk<T>(data.data(), i_data.data(), data.data() + shift, i_data.data() + shift, K);

    std::sort(buff.rbegin(), buff.rend());
    for(int i = 0;i < K; ++i) {
        //cout << "buff:(" << buff[i].k << "," << buff[i].v << ") ";
        //cout << "merge:(" << data[i] << "," << i_data[i] << ")"
        //    << endl;
        EXPECT_EQ(buff[i].k, data[i]);
        //EXPECT_EQ(buff[i].v, i_data[i]);
    }
}

TEST(CPU, topk) {
    const size_t N = 128*128*64;
    const size_t K = 50;

    vector<float> data(N, 0);
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(1000., 5.);
    for(size_t i = 0; i < N; ++i) {
        //data[i] = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)) );
        //data[i] = rand() % 100 ;
        data[i] = distribution(generator);
    }
    MinHeap<float> heap(data.data(), K);
    top_k<float>(data.data(), N, K, heap);
    float* out_val = heap.data();
    size_t* out_idx = heap.index();

    sort(data.rbegin(), data.rend());

    vector<float> out(K, 0);
    memcpy(out.data(), out_val, sizeof(float)*K);
    sort(out.rbegin(), out.rend());

    for(size_t i = 0; i < K; ++i) {
        float err = fabs(out[i] - data[i]);
        ASSERT_LE(err, 1e-4);
    }
}


TEST(GPU, topk_base2) {
    //cudaSetDevice(0);
    const size_t N = 128*128*64;
    const size_t K = 128;

    float * d_in[2];
    float * d_out;
    vector<float> out(K, 0);
    vector<float> data(N, 0);
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(1000., 5.);
    for(size_t i = 0; i < N; ++i) {
        data[i] = distribution(generator);
    }

    CHECK_CUDA(cudaMalloc((void**)&d_in[0], sizeof(float) * N));
    CHECK_CUDA(cudaMalloc((void**)&d_in[1], sizeof(float) * N));
    CHECK_CUDA(cudaMalloc((void**)&d_out, sizeof(float) * K));


    CHECK_CUDA(cudaMemcpy(d_in[0], data.data(), sizeof(float)*N, cudaMemcpyHostToDevice));
    fastTopK<float>(d_in[0], d_in[1], N, K, d_out);
    CHECK_CUDA(cudaMemcpy(out.data(), d_out, sizeof(float)*K, cudaMemcpyDeviceToHost));


    sort(data.begin(), data.end(), std::greater<float>());
    sort(out.begin(), out.end(), std::greater<float>());
    for(size_t i = 0; i < K; ++i) {
        float err = fabs(out[i] - data[i]);
        ASSERT_LE(err, 1e-5);
    }

    CHECK_CUDA(cudaFree(d_in[0]));
    CHECK_CUDA(cudaFree(d_in[1]));
    CHECK_CUDA(cudaFree(d_out));
}


TEST(GPU, topk_non_base2) {
    //cudaSetDevice(0);
    const size_t N = 128*128*80;
    const size_t K = 100;

    float * d_in[2];
    float * d_out;
    vector<float> out(K, 0);
    vector<float> data(N, 0);
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(1000., 5.);
    for(size_t i = 0; i < N; ++i) {
        data[i] = distribution(generator);
    }

    CHECK_CUDA(cudaMalloc((void**)&d_in[0], sizeof(float) * N));
    CHECK_CUDA(cudaMalloc((void**)&d_in[1], sizeof(float) * N));
    CHECK_CUDA(cudaMalloc((void**)&d_out, sizeof(float) * K));


    CHECK_CUDA(cudaMemcpy(d_in[0], data.data(), sizeof(float)*N, cudaMemcpyHostToDevice));
    fastTopK<float>(d_in[0], d_in[1], N, K, d_out);
    CHECK_CUDA(cudaMemcpy(out.data(), d_out, sizeof(float)*K, cudaMemcpyDeviceToHost));


    sort(data.begin(), data.end(), std::greater<float>());
    sort(out.begin(), out.end(), std::greater<float>());
    for(size_t i = 0; i < K; ++i) {
        float err = fabs(out[i] - data[i]);
        ASSERT_LE(err, 1e-5);
    }

    CHECK_CUDA(cudaFree(d_in[0]));
    CHECK_CUDA(cudaFree(d_in[1]));
    CHECK_CUDA(cudaFree(d_out));
}


TEST(GPU, bitonic_batch_topk) {
    const int batch_num = 1;
    int c = rand() % 100 + 1;
    const size_t slice_len = 128*128* c;
    const size_t N = batch_num * slice_len;
    const int K = 100;
    cout << "test size: 128*128*" << c
         << " batch size:1" << endl;

    typedef float T;

    size_t padding_num = (slice_len +255)/256 * 256;

    T * d_in;
    size_t * d_ind;
    vector<T> out(K*batch_num, 0);
    vector<size_t> ind(K*batch_num, 0);

    vector<T> data(N, 0);
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(10., 5.);
    for(size_t i = 0; i < N; ++i) {
        data[i] = static_cast<T>(distribution(generator));
    }

    CHECK_CUDA(cudaMalloc((void**)&d_in, sizeof(T) * padding_num * batch_num));
    CHECK_CUDA(cudaMalloc((void**)&d_ind, sizeof(size_t) * padding_num * batch_num));

    if (N % 256 == 0) {
        CHECK_CUDA(cudaMemcpy(d_in, data.data(), sizeof(T)*N, cudaMemcpyHostToDevice));
    }
    else {
        for (int i = 0;i < batch_num; ++i) {
            CHECK_CUDA(cudaMemcpy(d_in + padding_num * i, data.data() + slice_len * i, \
                        sizeof(T) * slice_len, cudaMemcpyHostToDevice));
        }
    }

    bitonicBatchTopK<T>(d_in, d_ind, batch_num, slice_len, K, NULL);
    CHECK_CUDA(cudaMemcpy(out.data(), d_in, sizeof(T)*K*batch_num, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ind.data(), d_ind, sizeof(size_t)*K*batch_num, cudaMemcpyDeviceToHost));

    for(int i = 0; i < batch_num; ++i) {
        auto lo = data.begin() + i * slice_len;
        sort(lo, lo+slice_len, std::greater<T>());
    }
    

    /*
    for(int i = 1; i < batch_num; ++i) {
        auto lo =  i * slice_len; 
        auto last = i * K;
        for(int j = 0; j < K; ++j) {
            data[last+j] = data[lo+j];
        }
    }
    */

    for(size_t i = 0; i < K*batch_num; ++i) {
        float err = fabs(out[i] - data[i]);
        if (err > 1e-5) cout << "\ni:" << i << " out:" << out[i] << " data:" << data[i] << endl;
        ASSERT_LE(err, 1e-5);
    }

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_ind));
}

TEST(GPU, bitonic_batch_topk2) {
    const int batch_num = rand() % 100 + 1;
    const size_t slice_len = 128*128;
    const size_t N = batch_num * slice_len;
    const int K = 100;
    cout << "test size: 128*128" 
         << " batch size:" 
         << batch_num << endl;

    typedef float T;

    size_t padding_num = (slice_len +255)/256 * 256;

    T * d_in;
    size_t * d_ind;
    vector<T> out(N, 0);
    vector<size_t> ind(N, 0);

    vector<T> data(N, 0);
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(10., 2);
    for(size_t i = 0; i < N; ++i) {
        data[i] = static_cast<T>(distribution(generator));
    }

    CHECK_CUDA(cudaMalloc((void**)&d_in, sizeof(T) * padding_num * batch_num));
    CHECK_CUDA(cudaMalloc((void**)&d_ind, sizeof(size_t) * padding_num * batch_num));

    CHECK_CUDA(cudaMemcpy(d_in, data.data(), sizeof(T)*N, cudaMemcpyHostToDevice));

    bitonicBatchTopK<T>(d_in, d_ind, batch_num, slice_len, K, NULL);
    CHECK_CUDA(cudaMemcpy(out.data(), d_in, sizeof(T)*N, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ind.data(), d_ind, sizeof(size_t)*N, cudaMemcpyDeviceToHost));

    for(int i = 0; i < batch_num; ++i) {
        auto lo = data.begin() + i * slice_len;
        sort(lo, lo+slice_len, std::greater<T>());
    }
    

    /*
    for(int i = 1; i < batch_num; ++i) {
        auto lo =  i * slice_len; 
        auto last = i * K;
        for(int j = 0; j < K; ++j) {
            data[last+j] = data[lo+j];
        }
    }
    */

    for(int n = 0; n < batch_num; ++n) {
        for(size_t i = 0; i < K; ++i) {
            float err = fabs(out[n*slice_len+i] - data[n*slice_len + i]);
            if (err > 1e-5) cout << "\nbatch id:"
                << n << " index:"
                << i << " out:" << out[n*slice_len+i] 
                << " data:" << data[n*slice_len+i] << endl;
            ASSERT_LE(err, 1e-5);
        }
    }

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_ind));
}



TEST(GPU, topk_non_base2_1) {
    const size_t N = 128*128*100;
    const size_t K = 100;

    float * d_in[2];
    float * d_out;
    vector<float> out(K, 0);
    vector<float> data(N, 0);
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(1000., 5.);
    for(size_t i = 0; i < N; ++i) {
        data[i] = distribution(generator);
    }

    CHECK_CUDA(cudaMalloc((void**)&d_in[0], sizeof(float) * N));
    CHECK_CUDA(cudaMalloc((void**)&d_in[1], sizeof(float) * N));
    CHECK_CUDA(cudaMalloc((void**)&d_out, sizeof(float) * K));


    CHECK_CUDA(cudaMemcpy(d_in[0], data.data(), sizeof(float)*N, cudaMemcpyHostToDevice));
    fastTopK<float>(d_in[0], d_in[1], N, K, d_out);
    CHECK_CUDA(cudaMemcpy(out.data(), d_out, sizeof(float)*K, cudaMemcpyDeviceToHost));


    sort(data.begin(), data.end(), std::greater<float>());
    sort(out.begin(), out.end(), std::greater<float>());
    for(size_t i = 0; i < K; ++i) {
        float err = fabs(out[i] - data[i]);
        ASSERT_LE(err, 1e-5);
    }

    CHECK_CUDA(cudaFree(d_in[0]));
    CHECK_CUDA(cudaFree(d_in[1]));
    CHECK_CUDA(cudaFree(d_out));
}


TEST(CPU, log2_series) {
    //vector<size_t> data = {128*128*80, 128*128*79, 128*128*30, 128*128*10, 79, 53, 530*717};
    for(int i = 0; i < 100; ++i) {
        vector<int> param;
        size_t data = rand();
        log2_series(data, param);
       // cout << " value: " << data[i] << " , log2_series:" << endl;
        size_t tmp = 0;
        for(auto &e: param) {
            //cout << e << " ";
            tmp += e >= 1 ? 2 << (e-1) : 1;
        }
        //cout << endl;
        //cout << " sum:" << tmp << " diff:" << data[i] - tmp << endl;
        EXPECT_EQ(data, tmp);
    }

    //vector<int> s;
    //log2_series(128*128*80, s);
    //cout << " log2_series(128*128*80):" << endl;
    //for(auto e : s) cout << e << " ";
    //cout << endl;
}



int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    srand (time(NULL));
    return RUN_ALL_TESTS();
}
 

