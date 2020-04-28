#include <gtest/gtest.h>
#include <vector>
#include "ResizeBilinear.hpp"

#include <stdlib.h> // rand
#include <time.h> 

#include <cuda_runtime_api.h> // cuda
#include <ATen/ATen.h> // libtorch, cpu version 

using namespace std;

#pragma pack(1)
struct Buffer {
    bool align_corners;
    float scales[2];
    int inDims[3];
    int outDims[3];
    Buffer(vector<float>& scales, bool align_corners,
            vector<int>& inDims, vector<int>& outDims) {
        this->align_corners = align_corners;
        this->scales[0] = scales[0];
        this->scales[1] = scales[1];
        for(int i = 0; i < 3; ++i) {
            this->inDims[i] = inDims[i];
            this->outDims[i] = outDims[i];
        }
    }
    const void* data() const {
        return this;
    }
    size_t size() const {
        return sizeof(bool) + sizeof(float)*2 + sizeof(int) * 6;
    }
};

#define cudaCheckError(e) { if(e != cudaSuccess) { \
    printf("cuda failure: %s:%d: '%s'\n", __FILE__, __LINE__, \
            cudaGetErrorString(e)); \
        exit(0); \
    } \
}

TEST(Foo, test) {
   const int a = 1;
   ASSERT_EQ(1, a);
}



TEST(MyUpsample, constructor) {
    vector<float> scale ({2.5, 3.2});
    bool align_corners = true;
    MyUpsamplePlugin tmp1(scale, align_corners);
    vector<int> inDims({3, 20, 20});
    vector<int> outDims({3, 50, 64});
    auto buffer = Buffer(scale, align_corners, inDims, outDims);
    MyUpsamplePlugin tmp2(buffer.data(), buffer.size());
    
    ASSERT_EQ(tmp1.align_corners(), tmp2.align_corners());
    ASSERT_EQ(tmp1.scales()[0], tmp2.scales()[0]);
    ASSERT_EQ(tmp1.scales()[1], tmp2.scales()[1]);
}


TEST(MyUpsample, clone) {
    vector<float> scale ({2.5, 3.2});
    bool align_corners = true;
    MyUpsamplePlugin tmp1(scale, align_corners);
    auto tmp2 = static_cast<MyUpsamplePlugin*>(tmp1.clone());

    ASSERT_EQ(tmp1.align_corners(), tmp2->align_corners());
    ASSERT_EQ(tmp1.scales()[0], tmp2->scales()[0]);
    ASSERT_EQ(tmp1.scales()[1], tmp2->scales()[1]);

    if(tmp2) delete tmp2;
}

TEST(MyUpsamplePluginCreator, deserializePlugin) {

}

TEST(MyUpsample, getOutputDimensions) {
    vector<float> scale(2, 0);
    scale[0] = static_cast<float>(rand() % 10) / 2.0 + 1;
    scale[1] = static_cast<float>(rand() % 10) / 2.0 + 1;
    bool align_corners = true;
    MyUpsamplePlugin tmp1(scale, align_corners);
    nvinfer1::Dims inputDims{3};
    const int C = rand() % 3 + 1;
    const int H = rand() % 1024 + 1;
    const int W = rand() % 1024 + 1;
    inputDims.d[0] = C;  
    inputDims.d[1] = H;
    inputDims.d[2] = W;
    auto outDims = tmp1.getOutputDimensions(0, &inputDims, 1);
    ASSERT_EQ(3, outDims.nbDims);
    ASSERT_EQ(C, outDims.d[0]);
    ASSERT_EQ(int(inputDims.d[1]*scale[0]), outDims.d[1]);
    ASSERT_EQ(int(inputDims.d[2]*scale[1]), outDims.d[2]);
}

TEST(MyUpsample, enqueue) {
    const int batch_size = rand() % 32 + 1;
    const int channels = rand() % 3 + 1;
    const int input_h = rand() % 224 + 1;
    const int input_w = rand() % 224 + 1;

    vector<float> scale(2);
    scale[0] = static_cast<float>(rand() % 10) / 2.0 + 1.;
    scale[1] = static_cast<float>(rand() % 10) / 2.0 + 1.;
    const int output_h = int(input_h * scale[0]);
    const int output_w = int(input_w * scale[1]);
    cout << "Testing enqueue kernel, inputs'shape: (" 
         << batch_size << ","
         << channels << ","
         << input_h << ","
         << input_w << ")" << endl;
    cout << " outputs'shape: (" 
         << batch_size << ","
         << channels << ","
         << output_h << ","
         << output_w << ")" << endl;
    int tmp = rand() % 10 + 1;
    bool align_corners = tmp > 5 ? true : false;
    MyUpsamplePlugin my_plugin(scale, align_corners);
    nvinfer1::Dims inputDims;
    inputDims.nbDims = 3;
    inputDims.d[0] = channels;
    inputDims.d[1] = input_h;  
    inputDims.d[2] = input_w;
    cout << "inDims:(";
    for(int i = 0; i < inputDims.nbDims; ++i) cout << inputDims.d[i] << ",";
    cout << ")" << endl;
    auto outDims = my_plugin.getOutputDimensions(0, &inputDims, 1);
    cout << "outDims:(";
    for(int i = 0; i < outDims.nbDims; ++i) cout << outDims.d[i] << ",";
    cout << ")" << endl;

    at::Tensor input_tensor = at::randn({batch_size, channels, input_h, input_w});

    at::Tensor out_tensor = at::zeros({batch_size, channels, output_h, output_w});
    void* buffers[2];
    void* workspace; 
    cudaCheckError(cudaSetDevice(0));
    cudaStream_t stream;
    cudaCheckError(cudaStreamCreate(&stream));
    cudaCheckError(cudaMalloc(&buffers[0], sizeof(float)*input_tensor.numel()));
    cudaCheckError(cudaMalloc(&buffers[1], sizeof(float)*out_tensor.numel()));
    cudaCheckError(cudaMemcpyAsync(buffers[0], input_tensor.data_ptr(), sizeof(float)* input_tensor.numel(), cudaMemcpyHostToDevice, stream));

    int flag = my_plugin.enqueue(batch_size, &buffers[0], &buffers[1], workspace, stream);
    cudaCheckError(cudaStreamSynchronize(stream));
    cudaCheckError(cudaMemcpy(out_tensor.data_ptr(), buffers[1], out_tensor.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    for(void * ptr : buffers) cudaFree(ptr);

    at::Tensor out_tensor2 = at::upsample_bilinear2d(input_tensor, {output_h, output_w}, align_corners);
    auto size = out_tensor2.sizes();

    auto ptr1 = static_cast<const float*>(out_tensor.data_ptr());
    auto ptr2 = static_cast<const float*>(out_tensor2.data_ptr());
    // verify
    for(size_t i = 0; i < out_tensor2.numel(); ++i) {
       float err = fabs(ptr1[i] - ptr2[i]);
       ASSERT_LE(err, 1e-5);
    }

}


TEST(MyUpsample, configurePlugin) {
    const int batch_size = rand() % 128 + 1;
    const int channels = rand() % 3 + 1;
    const int input_h = rand() % 224 + 1;
    const int input_w = rand() % 224 + 1;

    vector<float> scale(2);
    scale[0] = static_cast<float>(rand() % 10) / 2.0 + 1.;
    scale[1] = static_cast<float>(rand() % 10) / 2.0 + 1.;
    const int output_h = int(input_h * scale[0]);
    const int output_w = int(input_w * scale[1]);
    cout << "Testing configurePlugin kernel, inputs'shape: (" 
         << batch_size << ","
         << channels << ","
         << input_h << ","
         << input_w << ")" << endl;
    cout << " outputs'shape: (" 
         << batch_size << ","
         << channels << ","
         << output_h << ","
         << output_w << ")" << endl;
    int tmp = rand() % 10 + 1;
    bool align_corners = tmp > 5 ? true : false;
    MyUpsamplePlugin my_plugin(scale, align_corners);
    nvinfer1::Dims inputDims{3};
    inputDims.d[0] = channels;
    inputDims.d[1] = input_h;  inputDims.d[2] = input_w;
    nvinfer1::Dims outputDims{3};
    outputDims.d[0] = channels;
    outputDims.d[1] = output_h;  outputDims.d[2] = output_w;
    const bool inputIsBroadcast = false; 
    const bool outputIsBroadcast = false;
    const nvinfer1::DataType inputType = nvinfer1::DataType::kFLOAT;
    const nvinfer1::DataType outputType = nvinfer1::DataType::kFLOAT;

    my_plugin.configurePlugin(static_cast<const nvinfer1::Dims*>(&inputDims), 1, static_cast<const nvinfer1::Dims*>(&outputDims), 1,
            &inputType, 
            &outputType, 
            &inputIsBroadcast, 
            &outputIsBroadcast,
            nvinfer1::PluginFormat::kNCHW,
            32 
            );
    at::Tensor input_tensor = at::randn({batch_size, channels, input_h, input_w});

    at::Tensor out_tensor = at::zeros({batch_size, channels, output_h, output_w});
    void* buffers[2];
    void* workspace; 
    cudaCheckError(cudaSetDevice(0));
    cudaStream_t stream;
    cudaCheckError(cudaStreamCreate(&stream));
    cudaCheckError(cudaMalloc(&buffers[0], sizeof(float)*input_tensor.numel()));
    cudaCheckError(cudaMalloc(&buffers[1], sizeof(float)*out_tensor.numel()));
    cudaCheckError(cudaMemcpyAsync(buffers[0], input_tensor.data_ptr(), sizeof(float)* input_tensor.numel(), cudaMemcpyHostToDevice, stream));

    int flag = my_plugin.enqueue(batch_size, &buffers[0], &buffers[1], workspace, stream);
    cudaCheckError(cudaStreamSynchronize(stream));
    cudaCheckError(cudaMemcpy(out_tensor.data_ptr(), buffers[1], out_tensor.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    for(void * ptr : buffers) cudaFree(ptr);

    at::Tensor out_tensor2 = at::upsample_bilinear2d(input_tensor, {output_h, output_w}, align_corners);
    auto size = out_tensor2.sizes();

    auto ptr1 = static_cast<const float*>(out_tensor.data_ptr());
    auto ptr2 = static_cast<const float*>(out_tensor2.data_ptr());
    // verify
    for(size_t i = 0; i < out_tensor2.numel(); ++i) {
       float err = fabs(ptr1[i] - ptr2[i]);
       ASSERT_LE(err, 1e-5);
    }
}



int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    srand (time(NULL));
    return RUN_ALL_TESTS();
}
