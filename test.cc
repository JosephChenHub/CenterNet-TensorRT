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
#include "ctdet_kernels.hpp"
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
    float LO = 0;
    float HI = 65534;
    std::default_random_engine generator;
    std::normal_distribution<T> distribution(1000., 5.);
    for(size_t i = 0; i < N; ++i) {
        //data[i] = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)) );
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
    float LO = 0;
    float HI = 65534;
    //srand(time(NULL));
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

TEST(GPU, mergeSort_base2) {
    cudaSetDevice(0);
    const size_t N = 128*128*64;
    typedef float T;

    T * d_in;
    T * d_out;
    vector<T> out(N, 0);
    vector<T> data(N, 0);
    float LO = 1;
    float HI = 65000;
    std::default_random_engine generator;
    std::normal_distribution<T> distribution(1000., 5.);
    for(size_t i = 0; i < N; ++i) {
        //data[i] = LO + static_cast <T> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)) );
        //data[i] = rand() % 500 + 1;
        data[i] = distribution(generator);
    }

    //cout << "input:\n";
    //for(size_t i = 0; i < N; ++i) cout << data[i] << " ";
    //cout << endl;

    CHECK_CUDA(cudaMalloc((void**)&d_in,  sizeof(T) * N));
    CHECK_CUDA(cudaMalloc((void**)&d_out, sizeof(T) * N));
    CHECK_CUDA(cudaMemcpy(d_in,  data.data(), sizeof(T)*N, cudaMemcpyHostToDevice));
    mergeSort<T>(d_in, N, d_out);
    CHECK_CUDA(cudaMemcpy(out.data(), d_in, sizeof(T)*N,  cudaMemcpyDeviceToHost));

    //cout << " rank:" << endl;
    //for(int i = 0; i < 4; ++i) {
    //    for(int  j =0; j < 256; ++j) cout << out[j + 256*i] << " ";
    //    cout << " ##\n";
    //}
    //cout <<" ~~~~~~~~~~~" << endl;

    sort(data.begin(), data.end());
    for(size_t i = 0; i < N; ++i) {
        float err = fabs(out[i] - data[i]);
        ASSERT_LE(err, 1e-5);
    }
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}
TEST(GPU, mergeSort_non_base2) {
    //cudaSetDevice(0);
    const size_t N = 1377 * 777;
    typedef float T;

    T * d_in;
    T * d_buff;

    size_t padding_num = (N + 255) / 256;
    padding_num *= 256;

    vector<T> out(N, 0);
    vector<T> data(N, 0);
    float LO = 0;
    float HI = 65000;
    std::default_random_engine generator;
    std::normal_distribution<T> distribution(1000., 5.);
    for(size_t i = 0; i < N; ++i) {
        //data[i] = LO + static_cast <T> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)) );
        //data[i] = rand() % 500 + 1;
        data[i] = distribution(generator);
    }

    //cout << "input:\n";
    //for(size_t i = 0; i < N; ++i) cout << data[i] << " ";
    //cout << endl;

    CHECK_CUDA(cudaMalloc((void**)&d_in,  sizeof(T) * padding_num));
    CHECK_CUDA(cudaMalloc((void**)&d_buff, sizeof(T) * padding_num));
    CHECK_CUDA(cudaMemcpy(d_in,  data.data(), sizeof(T)*N, cudaMemcpyHostToDevice));
    mergeSort<T>(d_in, N, d_buff);
    CHECK_CUDA(cudaMemcpy(out.data(), d_in, sizeof(T)*N,  cudaMemcpyDeviceToHost));

    //cout << " rank:" << endl;
    //for(int i = 0; i < 4; ++i) {
    //    for(int  j =0; j < 256; ++j) cout << out[j + 256*i] << " ";
    //    cout << " ##\n";
    //}
    //cout <<" ~~~~~~~~~~~" << endl;

    sort(data.begin(), data.end());
    for(size_t i = 0; i < N; ++i) {
        float err = fabs(out[i] - data[i]);
        ASSERT_LE(err, 1e-5);
    }
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_buff));
}

TEST(GPU, mergeSort_non_base2_down) {
    const size_t N = 128*128*64+256;//   1377 * 777;
    typedef float T;

    T * d_in;
    T * d_buff;
    vector<T> out(N, 0);
    vector<T> data(N, 0);
    float LO = 0;
    float HI = 65000;
    std::default_random_engine generator;
    std::normal_distribution<T> distribution(1000., 5.);
    for(size_t i = 0; i < N; ++i) {
        //data[i] = LO + static_cast <T> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)) );
        //data[i] = rand() % 65000 + 1;
        data[i] = distribution(generator);
    }

    size_t num_blocks = (N + 255) / 256;

    CHECK_CUDA(cudaMalloc((void**)&d_in,  sizeof(T) * num_blocks * 256));
    CHECK_CUDA(cudaMalloc((void**)&d_buff, sizeof(T) * num_blocks * 256));
    CHECK_CUDA(cudaMemcpy(d_in,  data.data(), sizeof(T)*N, cudaMemcpyHostToDevice));
    mergeSort<T>(d_in, N, d_buff, false);
    CHECK_CUDA(cudaMemcpy(out.data(), d_in, sizeof(T)*N,  cudaMemcpyDeviceToHost));

    //cout << " rank:" << endl;
    //for(int i = 0; i < 4; ++i) {
    //    for(int  j =0; j < 256; ++j) cout << out[j + 256*i] << " ";
    //    cout << " ##\n";
    //}
    //cout <<" ~~~~~~~~~~~" << endl;

    sort(data.begin(), data.end(), std::greater<T>());
    for(size_t i = 0; i < N; ++i) {
        float err = fabs(out[i] - data[i]);
        if(err > 1e-5) {
            cout << "total num:" << data.size() ;
            cout <<" i:" << i << " data:" << data[i] << " out:" << out[i] << endl;
        }
        ASSERT_LE(err, 1e-5);
    }
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_buff));
}

TEST(GPU, topk_base2) {
    //cudaSetDevice(0);
    const size_t N = 128*128*64;
    const size_t K = 100;

    float * d_in[2];
    float * d_out;
    vector<float> out(K, 0);
    vector<float> data(N, 0);
    float LO = 0;
    float HI = 65000;
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(1000., 5.);
    for(size_t i = 0; i < N; ++i) {
        //data[i] =  static_cast<float>(rand() % 10);
        //data[i] = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)) );
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
    float LO = 1;
    float HI = 65000;
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(1000., 5.);
    for(size_t i = 0; i < N; ++i) {
        //data[i] =  static_cast<float>(rand() % 10);
        //data[i] = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)) );
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


/*
TEST(GPU, bitonic_batch_topk) {
    const int batch_num = 80;
    const size_t slice_len = 128*128;
    const size_t N = batch_num * slice_len;
    const size_t K = 100;

    typedef float T;

    size_t padding_num = (slice_len +255)/256 * 256;

    T * d_in;
    size_t * d_ind;
    vector<T> out(K*batch_num, 0);
    vector<size_t> ind(K*batch_num, 0);

    vector<T> data(N, 0);
    float LO = 1;
    float HI = 65000;
    vector<T> hi(batch_num);
    for (int i = 0;i  < batch_num; ++i) hi[i] = (i == 0) ? 30000: hi[i-1] + 10000;
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(1000., 5.);
    for(size_t i = 0; i < N; ++i) {
        //data[i] = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)) );
        //data[i] = i % hi[i / slice_len];
        data[i] = distribution(generator);
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

    bitonicBatchTopK<T>(d_in, d_ind, batch_num, slice_len, K);
    CHECK_CUDA(cudaMemcpy(out.data(), d_in, sizeof(T)*K*batch_num, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ind.data(), d_ind, sizeof(size_t)*K*batch_num, cudaMemcpyDeviceToHost));

    for(int i = 0; i < batch_num; ++i) {
        auto lo = data.begin() + i * slice_len;
        sort(lo, lo+slice_len, std::greater<float>());
    }
    

    for(int i = 1; i < batch_num; ++i) {
        auto lo =  i * slice_len; 
        auto last = i * K;
        for(int j = 0; j < K; ++j) {
            data[last+j] = data[lo+j];
        }
    }

    for(size_t i = 0; i < K*batch_num; ++i) {
        float err = fabs(out[i] - data[i]);
        if (err > 1e-5) cout << "\ni:" << i << " out:" << out[i] << " data:" << data[i] << endl;
        ASSERT_LE(err, 1e-5);
    }

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_ind));
}

TEST(GPU, bitonic_batch_topk2) {
    const int batch_num = 1;
    const size_t slice_len = 80 * 128*128;
    const size_t N = batch_num * slice_len;
    const size_t K = 100;

    typedef float T;

    size_t padding_num = (slice_len +255)/256 * 256;

    T * d_in;
    size_t * d_ind;
    vector<T> out(K*batch_num, 0);
    vector<size_t> ind(K*batch_num, 0);

    vector<T> data(N, 0);
    float LO = 1;
    float HI = 65000;
    vector<T> hi(batch_num);
    for (int i = 0;i  < batch_num; ++i) hi[i] = (i == 0) ? 30000: hi[i-1] + 10000;
    std::default_random_engine generator;
    std::normal_distribution<T> distribution(1000., 5.);
    for(size_t i = 0; i < N; ++i) {
        //data[i] = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)) );
        //data[i] = i % hi[i / slice_len];
        data[i] = distribution(generator);
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


    bitonicBatchTopK<T>(d_in, d_ind, batch_num, slice_len, K);
    CHECK_CUDA(cudaMemcpy(out.data(), d_in, sizeof(T)*K*batch_num, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ind.data(), d_ind, sizeof(size_t)*K*batch_num, cudaMemcpyDeviceToHost));

    for(int i = 0; i < batch_num; ++i) {
        auto lo = data.begin() + i * slice_len;
        sort(lo, lo+slice_len, std::greater<float>());
    }
    

    for(int i = 1; i < batch_num; ++i) {
        auto lo =  i * slice_len; 
        auto last = i * K;
        for(int j = 0; j < K; ++j) {
            data[last+j] = data[lo+j];
        }
    }

    for(size_t i = 0; i < K*batch_num; ++i) {
        float err = fabs(out[i] - data[i]);
        if (err > 1e-5) cout << "\ni:" << i << " out:" << out[i] << " data:" << data[i] << endl;
        ASSERT_LE(err, 1e-5);
    }

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_ind));
}
*/

TEST(GPU, ctdet_topk) {
    cudaSetDevice(0);
    const int batch_num = 1;
    const int channel = 80;
    const int height = 128;
    const int width  = 128;
    const size_t slice_len = channel * height * width;
    const size_t N = batch_num * slice_len;
    const size_t K = 100;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    


    size_t padding_num = (slice_len +255)/256 * 256;

    float * d_in;
    size_t * d_ind;
    vector<float> out(K*batch_num, 0);
    vector<size_t> ind(K*batch_num, 0);

    vector<float> data(N, 0);
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(1000., 10.0);
    for(size_t i = 0; i < N; ++i) {
        data[i] = distribution(generator);
    }
    vector<float> h_wh(batch_num * 2 * height * width);
    vector<float> h_reg(batch_num * 2 * height * width); 

    for(size_t i =  0; i < h_wh.size(); ++i) {
        h_wh[i] = distribution(generator);
        h_reg[i] = distribution(generator);
    }

    float* d_bbox;
    float* d_wh;
    float* d_reg;

    CHECK_CUDA(cudaMalloc((void**)&d_in, sizeof(float) * padding_num * batch_num));
    CHECK_CUDA(cudaMalloc((void**)&d_ind, sizeof(size_t) * padding_num * batch_num));
    CHECK_CUDA(cudaMalloc((void**)&d_wh, sizeof(float) * h_wh.size()));
    CHECK_CUDA(cudaMalloc((void**)&d_reg, sizeof(float) * h_reg.size()));
    CHECK_CUDA(cudaMalloc((void**)&d_bbox, sizeof(float) * 6 * K * batch_num));

    if (N % 256 == 0) {
        CHECK_CUDA(cudaMemcpy(d_in, data.data(), sizeof(float)*N, cudaMemcpyHostToDevice));
    }
    else {
        for (int i = 0;i < batch_num; ++i) {
            CHECK_CUDA(cudaMemcpy(d_in + padding_num * i, data.data() + slice_len * i, \
                        sizeof(float) * slice_len, cudaMemcpyHostToDevice));
        }
    }
    CHECK_CUDA(cudaMemcpy(d_wh, h_wh.data(), sizeof(float)*h_wh.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_reg, h_reg.data(),
                sizeof(float)*h_reg.size(), cudaMemcpyHostToDevice));

    float center[] = {540, 821};
    float scale[] = {1642, 1642};
    ctdet_decode(d_bbox, d_wh, d_reg, 
            (float*)d_in, d_ind,  
            center, scale, 
            batch_num, \
            channel, height, width, K, 80, 0.3,true, false, stream);

    CHECK_CUDA(cudaMemcpy(out.data(), d_in, sizeof(float)*K*batch_num, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ind.data(), d_ind, sizeof(size_t)*K*batch_num, cudaMemcpyDeviceToHost));

    vector<pair<float, size_t>> buff;
    for(int i = 0; i < batch_num; ++i) {
        //auto lo = data.begin() + i * slice_len;
        //sort(lo, lo+slice_len, std::greater<float>());
        for(size_t j =  0; j < slice_len; ++j) {
            auto tmp = make_pair(data[slice_len* i + j], j);
            buff.push_back(tmp);
        } 
    }
    ASSERT_EQ(buff.size(), N);

    //for(int i = 0; i < batch_num; ++i) {
    //    auto lo = data.begin() + i * slice_len;
    //    sort(lo, lo+slice_len, std::greater<float>());
    //}
    for(int i = 0; i < batch_num; ++i) {
        auto lo = buff.begin() + i * slice_len;
        sort(lo, lo+slice_len, [&](const pair<float, size_t>& a, const pair<float, size_t>& b)->bool {return a.first > b.first; } );
    }

    for(size_t i = 1; i < batch_num; ++i) {
        auto lo =  i * slice_len; 
        auto last = i * K;
        for(size_t j = 0; j < K; ++j) {
            //data[last+j] = data[lo+j];
            buff[last+j] = buff[lo+j];
        }
    }

    for(size_t i = 0; i < K*batch_num; ++i) {
        float err = fabs(out[i] - buff[i].first);
        if (err > 1e-5) cout << "\ndata-i:" << i << " out:" << out[i] << " buff:" << buff[i].first << endl;
        ASSERT_LE(err, 1e-5);
    }

    for(size_t i = 0; i < K*batch_num; ++i) {
        float err = fabs(ind[i] - buff[i].second % (128*128));
        if (err > 1e-5) cout << "\ni:" << i << " ind:" << ind[i] << " buff:" << buff[i].second << endl;
        ASSERT_LE(err, 1e-5);
    }



    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_ind));
    CHECK_CUDA(cudaFree(d_reg));
    CHECK_CUDA(cudaFree(d_wh));
    CHECK_CUDA(cudaFree(d_bbox));
    cudaStreamDestroy(stream);
}

TEST(GPU, topk_non_base2_1) {
    const size_t N = 128*128*100;
    const size_t K = 100;

    float * d_in[2];
    float * d_out;
    vector<float> out(K, 0);
    vector<float> data(N, 0);
    float LO = 0;
    float HI = 65000;
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(1000., 5.);
    for(size_t i = 0; i < N; ++i) {
        //data[i] =  static_cast<float>(rand() % 10);
        //data[i] = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)) );
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

TEST(GPU, bitonicSort) {
    const size_t N = 128*128*64;
    typedef float T;

    T * d_in;
    T * d_out;
    vector<T> out(N, 0);
    vector<T> data(N, 0);
    float LO = 1;
    float HI = 65000;
    std::default_random_engine generator;
    std::normal_distribution<T> distribution(1000., 5.);
    for(size_t i = 0; i < N; ++i) {
        //data[i] = LO + static_cast <T> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)) );
        //data[i] = rand() % 1024;
        data[i] = distribution(generator);
    }

    //cout << "input:\n";
    //for(size_t i = 0; i < N; ++i) cout << data[i] << " ";
    //cout << endl;

    CHECK_CUDA(cudaMalloc((void**)&d_in,  sizeof(T) * N));
    CHECK_CUDA(cudaMalloc((void**)&d_out, sizeof(T) * N));
    CHECK_CUDA(cudaMemcpy(d_in,  data.data(), sizeof(T)*N, cudaMemcpyHostToDevice));
    bitonicSort<T>(d_in, N, d_out);
    CHECK_CUDA(cudaMemcpy(out.data(), d_in, sizeof(T)*N,  cudaMemcpyDeviceToHost));

    //cout << " rank:" << endl;
    //for(int i = 0; i < 4; ++i) {
    //    for(int  j =0; j < 256; ++j) cout << out[j + 256*i] << " ";
    //    cout << " ##\n";
    //}
    //cout <<" ~~~~~~~~~~~" << endl;
    //cout << " raw data:" << endl;
    //for(int i = 0; i < 4; ++i) {
    //    for(int  j =0; j < 256; ++j) cout << data[j + 256*i] << " ";
    //    cout << " ##\n";
    //}
    //cout <<" ==============" << endl;

    sort(data.begin(), data.end());
    for(size_t i = 0; i < N; ++i) {
        float err = fabs(out[i] - data[i]);
        if (err > 1e-5) cout << "i:" << i << " out:" << out[i] << " data:" << data[i] << endl;
        ASSERT_LE(err, 1e-5);
    }
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}

TEST(GPU, bitonicSort_non_base2) {
    const size_t N = 1377 * 777;
    typedef float T;

    T * d_in;
    T * d_out;
    vector<T> out(N, 0);
    vector<T> data(N, 0);
    float LO = 0;
    float HI = 65000;
    std::default_random_engine generator;
    std::normal_distribution<T> distribution(1000., 5.);
    for(size_t i = 0; i < N; ++i) {
        //data[i] = LO + static_cast <T> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)) );
        //data[i] = rand() % 1024;
        data[i] = distribution(generator);
    }

    //cout << "input:\n";
    //for(size_t i = 0; i < N; ++i) cout << data[i] << " ";
    //cout << endl;

    size_t padding_num = (N+255)/256;
    padding_num *= 256;

    CHECK_CUDA(cudaMalloc((void**)&d_in,  sizeof(T) * padding_num));
    CHECK_CUDA(cudaMalloc((void**)&d_out, sizeof(T) * padding_num));
    CHECK_CUDA(cudaMemcpy(d_in,  data.data(), sizeof(T)*N, cudaMemcpyHostToDevice));
    bitonicSort<T>(d_in, N, d_out);
    CHECK_CUDA(cudaMemcpy(out.data(), d_in, sizeof(T)*N,  cudaMemcpyDeviceToHost));

    //cout << " rank:" << endl;
    //for(int i = 0; i < 4; ++i) {
    //    for(int  j =0; j < 256; ++j) cout << out[j + 256*i] << " ";
    //    cout << " ##\n";
    //}
    //cout <<" ~~~~~~~~~~~" << endl;
    //cout << " raw data:" << endl;
    //for(int i = 0; i < 4; ++i) {
    //    for(int  j =0; j < 256; ++j) cout << data[j + 256*i] << " ";
    //    cout << " ##\n";
    //}
    //cout <<" ==============" << endl;

    sort(data.begin(), data.end());
    for(size_t i = 0; i < N; ++i) {
        float err = fabs(out[i] - data[i]);
        if (err > 1e-5) cout << "i:" << i << " out:" << out[i] << " data:" << data[i] << endl;
        ASSERT_LE(err, 1e-5);
    }
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}

TEST(GPU, bitonicSort_descending) {
    const size_t N = 128*128*64;
    typedef float T;

    T * d_in;
    T * d_out;
    vector<T> out(N, 0);
    vector<T> data(N, 0);
    float LO = 0;
    float HI = 65000;
    std::default_random_engine generator;
    std::normal_distribution<T> distribution(1000., 5.);
    for(size_t i = 0; i < N; ++i) {
        //data[i] = LO + static_cast <T> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)) );
        //data[i] = rand() % 1024;
        data[i] = distribution(generator);
    }

    //cout << "input:\n";
    //for(size_t i = 0; i < N; ++i) cout << data[i] << " ";
    //cout << endl;

    CHECK_CUDA(cudaMalloc((void**)&d_in,  sizeof(T) * N));
    CHECK_CUDA(cudaMalloc((void**)&d_out, sizeof(T) * N));
    CHECK_CUDA(cudaMemcpy(d_in,  data.data(), sizeof(T)*N, cudaMemcpyHostToDevice));
    bitonicSort<T>(d_in, N, d_out, false);
    CHECK_CUDA(cudaMemcpy(out.data(), d_in, sizeof(T)*N,  cudaMemcpyDeviceToHost));

    //cout << " rank:" << endl;
    //for(int i = 0; i < 4; ++i) {
    //    for(int  j =0; j < 256; ++j) cout << out[j + 256*i] << " ";
    //    cout << " ##\n";
    //}
    //cout <<" ~~~~~~~~~~~" << endl;
    //cout << " raw data:" << endl;
    //for(int i = 0; i < 4; ++i) {
    //    for(int  j =0; j < 256; ++j) cout << data[j + 256*i] << " ";
    //    cout << " ##\n";
    //}
    //cout <<" ==============" << endl;

    sort(data.begin(), data.end(), std::greater<T>());
    for(size_t i = 0; i < N; ++i) {
        float err = fabs(out[i] - data[i]);
        if (err > 1e-5) cout << "i:" << i << " out:" << out[i] << " data:" << data[i] << endl;
        ASSERT_LE(err, 1e-5);
    }
    CHECK_CUDA(cudaFree(d_in));
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


TEST(GPU, mergeSort_pair) {
    const size_t N = 128*128*80;
    
    typedef Pair<float, size_t> T;

    T* d_data;
    T* d_buff;
    float LO = 1;
    float HI = 65000;

    vector<T> out(N);
    vector<T> h_data(N);
    for(size_t i = 0;i < N; ++i) {
        h_data[i].v = i;
        h_data[i].k = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)) );
    }

    //cout << "raw data:\n";
    //for(size_t i = 0; i < 256; ++i) cout << "(" << h_data[i].k << "," << h_data[i].v << ") ";
    //cout << endl;


    cudaMalloc((void**)&d_data, sizeof(T) * N);
    cudaMalloc((void**)&d_buff, sizeof(T) * N);
    cudaMemcpy(d_data, h_data.data(), sizeof(T)*N, cudaMemcpyHostToDevice);
    mergeSort<T>(d_data, N, d_buff);

    cudaMemcpy(out.data(), d_data, sizeof(T)*N, cudaMemcpyDeviceToHost);
    //cout << "sorted data:\n";
    //for(size_t i = 0; i < 256; ++i) cout << "(" << out[i].k << "," << out[i].v << ") ";
    //cout << endl;

    sort(h_data.begin(), h_data.end());

    for(size_t i = 0; i < N; ++i) {
        ASSERT_EQ(h_data[i], out[i]);
    }
    
    cudaFree(d_data);
    cudaFree(d_buff);
}

TEST(GPU, bitonicSort_pair) {
    const size_t N = 128*128*80;
    
    typedef Pair<float, size_t> T;

    T* d_data;
    T* d_buff;
    float LO = 1;
    float HI = 65000;

    vector<T> out(N);
    vector<T> h_data(N);
    for(size_t i = 0;i < N; ++i) {
        h_data[i].v = i;
        h_data[i].k = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)) );
    }

    //cout << "raw data:\n";
    //for(size_t i = 0; i < 256; ++i) cout << "(" << h_data[i].k << "," << h_data[i].v << ") ";
    //cout << endl;


    cudaMalloc((void**)&d_data, sizeof(T) * N);
    cudaMalloc((void**)&d_buff, sizeof(T) * N);
    cudaMemcpy(d_data, h_data.data(), sizeof(T)*N, cudaMemcpyHostToDevice);
    bitonicSort<T>(d_data, N, d_buff);

    cudaMemcpy(out.data(), d_data, sizeof(T)*N, cudaMemcpyDeviceToHost);
    //cout << "sorted data:\n";
    //for(size_t i = 0; i < 256; ++i) cout << "(" << out[i].k << "," << out[i].v << ") ";
    //cout << endl;

    sort(h_data.begin(), h_data.end());

    for(size_t i = 0; i < N; ++i) {
        ASSERT_EQ(h_data[i], out[i]);
    }
    
    cudaFree(d_data);
    cudaFree(d_buff);
}

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    srand (time(NULL));
    return RUN_ALL_TESTS();
}
 

