//#include <cuda_runtime_api.h> // cuda
#include <cuda_runtime.h>

#include "NvInfer.h"
#include <gtest/gtest.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstring>
#include <memory>
#include <math.h>
#include "dcn_v2.hpp"

#include <stdlib.h> // rand
#include <time.h> 
#include <random>


//#include <ATen/ATen.h> // libtorch, cpu version 

#include "common/logger.h"

using namespace std;
using namespace nvinfer1;


#define CHECK_CUDA(e) { if(e != cudaSuccess) { \
    printf("cuda failure: %s:%d: '%s'\n", __FILE__, __LINE__, \
            cudaGetErrorString(e)); \
        exit(0); \
    } \
}


#pragma pack(1)
struct DCNBuffer {
    int inDims[3];
    int outDims[3];
    int kernel_size;
    int dilation;
    int padding;
    int stride;
    int deformable_groups;
    DCNBuffer(std::vector<int> const& inDims, std::vector<int> const& outDims, const int kernel_size, const int dilation,
            const int padding, const int stride, 
            const int deformable_groups) {
        for(int i = 0; i < 3; ++i) {
            this->inDims[i] = inDims[i];
            this->outDims[i] = outDims[i];
        }
        this->kernel_size = kernel_size;
        this->dilation = dilation;
        this->padding = padding;
        this->stride = stride;
        this->deformable_groups = deformable_groups;
    }
    void * data() {
        return (void*)this;
    }
    size_t size() const {
        return sizeof(bool) + sizeof(float)*2 + sizeof(int) * 6;
    }
};
struct NvInferDeleter {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};
int getBindingInputIndex(IExecutionContext* context) {
    return !context->getEngine().bindingIsInput(0); // 0 (false) if bindingIsInput(0), 1 (true) otherwise
}
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



TEST(Foo, test) {
   const int a = 1;
   ASSERT_EQ(1, a);
}






TEST(DCNPluginCreator, deserializePlugin) {

}

TEST(DCN, getOutputDimensions) {
}

TEST(DCN, enqueue) {
    const int gpu_id = 0;
    CHECK_CUDA(cudaSetDevice(gpu_id));
    cudaStream_t stream; 
    CHECK_CUDA(cudaStreamCreate(&stream));    
    /// read serialized engine from local file 
    vector<char> trtModelStream_;
    size_t size{0};
    string file_name = "test_dcn.trt";
    cout << "Loading engine file:" << file_name << std::endl;
    ifstream file(file_name, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream_.resize(size);
        file.read(trtModelStream_.data(), size);
        file.close();
    }
    cout << " size: " << size << endl;
    auto runtime = unique_ptr<IRuntime, NvInferDeleter>(createInferRuntime(gLogger));
    assert(runtime);
    auto engine = unique_ptr<ICudaEngine, NvInferDeleter>(runtime->deserializeCudaEngine( \
            trtModelStream_.data(), size, nullptr));
    if(!engine) {
        cerr << " Failed to create the engine from .trt file!" << endl;
    }
    /// an execution context holds additional memory to store intermediate activation values. an engine can have multiple contexts sharing the same weights for multi-tasks/streams
    auto context = unique_ptr<IExecutionContext, NvInferDeleter>(engine->createExecutionContext());
    if(!context) {
        cerr << " Failed to createExecutionContext!" << endl;
        return ;
    }
    
    ifstream x_file("x_data.txt");
    ifstream y_file("y_data.txt");
    string line;
    vector<float> x_data;
    vector<float> y_data;
    while(getline(x_file, line)) {
        vector<string> tmp = split_str(line, ",");
        for(auto&e :tmp) x_data.push_back(atof(e.c_str()));
    }
    //std::cout << "x_data:" << x_data.size() << std::endl;
    //for(int i = 0; i < 5; ++i) cout << x_data[i] << " ";
    //cout << endl;
    while(getline(y_file, line)) {
        vector<string> tmp = split_str(line, ",");
        for(auto&e :tmp) y_data.push_back(atof(e.c_str()));
    }
    //std::cout << "y_data:" << y_data.size() << std::endl;
    //for(int i = 0; i < 5; ++i) cout << y_data[i] << " ";

    vector<size_t> in_shape {5, 3, 40, 40};
    vector<size_t> out_shape {5, 6, 40, 40 };
    size_t o_size = 1, i_size = 1;
    for ( int i = 0 ; i < 4; ++i ) {
        o_size *= out_shape[i];
        i_size *= in_shape[i];
    }
    float * output = new float[o_size];

    void* buffers[2];
    CHECK_CUDA(cudaMalloc(&buffers[0], sizeof(float)*i_size));
    CHECK_CUDA(cudaMalloc(&buffers[1], sizeof(float)*o_size));
    CHECK_CUDA(cudaMemcpyAsync(buffers[0], x_data.data(), sizeof(float)*i_size, cudaMemcpyHostToDevice, stream));
    context->enqueue(1, buffers, stream, nullptr);
    CHECK_CUDA(cudaMemcpyAsync(output, buffers[1], sizeof(float)*o_size, cudaMemcpyDeviceToHost, stream));

    //cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);


    // verify
    for(size_t i = 0; i < o_size; ++i) {
       float err = fabs(output[i] - y_data[i]);
       if (err > 1e-5) {
           int n = i / (o_size / out_shape[0]);
           int c = i % (o_size / out_shape[0]) / (out_shape[2] * out_shape[3]);
           cout << " batch_id:" << n << endl;
       }
       ASSERT_LE(err, 1e-5);
    }

    for(int i = 0; i < 2; ++i) CHECK_CUDA(cudaFree(buffers[i]));

    delete [] output;
}


TEST(DCN, configurePlugin) {
}



int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    srand (time(NULL));
    return RUN_ALL_TESTS();
}

