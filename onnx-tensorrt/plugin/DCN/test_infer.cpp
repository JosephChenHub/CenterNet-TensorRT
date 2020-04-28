#include "NvInfer.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <cstring>
#include <numeric>
#include <chrono> // test time
#include <opencv2/opencv.hpp>

#include "common/logger.h"
#include "ResizeBilinear.hpp"
#include "kernel.h"

#ifdef USE_LIBTORCH
#include <ATen/ATen.h>
#endif 

#define TEST_IMAGE

#define USE_PADDING // use padding to scale the input image

using namespace nvinfer1;
using namespace std;
using namespace cv;

struct NvInferDeleter {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};

inline int divUp(int a, int b) {
    return (a + b - 1) / b;
}

int getBindingInputIndex(IExecutionContext* context) {
    return !context->getEngine().bindingIsInput(0); // 0 (false) if bindingIsInput(0), 1 (true) otherwise
}

#ifdef TEST_IMAGE
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



int main(int argc, char* argv[]) {
    const int gpu_id = 0;
    cudaCheckError(cudaSetDevice(gpu_id));
    cudaStream_t stream; 
    cudaCheckError(cudaStreamCreate(&stream));    

    /// read serialized engine from local file 
    vector<char> trtModelStream_;
    size_t size{0};
    string file_name(argv[1]);
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
        return -1;
    }

    /// allocate memory for input&output 
    const int batchSize = 1;
    int inputIndex = getBindingInputIndex(context.get());
    int outputIndex = 1 - inputIndex; 
    cout << "inputIdx:" << inputIndex << endl;
    void* buffers[3];
    Dims inputDims{4}; 
    Dims outputDims{4};
    nvinfer1::DataType inputType, outputType;
    size_t inputVol, outputVol;

    for(int i = 0; i < engine->getNbBindings(); ++i) {
        auto dims = context ? context->getBindingDimensions(i):engine->getBindingDimensions(i);
        size_t vol = context ? 1 : static_cast<size_t>(batchSize);
        nvinfer1::DataType type = engine->getBindingDataType(i);
        int vecDim = engine->getBindingVectorizedDim(i);
        if(-1 != vecDim) {
            int scalarsPerVec = engine->getBindingComponentsPerElement(i);
            dims.d[vecDim] = divUp(dims.d[vecDim], scalarsPerVec);
            vol *= scalarsPerVec;
        }
        vol *= std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
        if(i == inputIndex) {
            inputDims = dims; 
            inputVol = vol;
            cout << "inputs' NCHW:" << dims.d[0] 
                << ", " << dims.d[1] << "," << dims.d[2]
                <<"," << dims.d[3]  << " dataType:" << sizeof(type) << endl;
        } else {
            outputDims = dims;
            outputVol = vol;
            cout << "outputs' NCHW:" << dims.d[0] 
                << ", " << dims.d[1] << "," << dims.d[2]
                <<"," << dims.d[3] << " dataType:" << sizeof(type) << endl;
        }
        // cuda mem.
        cudaMalloc(&buffers[i], vol * sizeof(type));
    }

    /// mem. host->dev.
    int N = inputDims.d[0], C = inputDims.d[1], H = inputDims.d[2], W = inputDims.d[3];
    cout << " NCHW:" << N << ", " << C << "," << H <<"," << W << endl;
    vector<float> mean({0.485, 0.456, 0.406});
    vector<float> std({0.229, 0.224, 0.225});
    //vector<float> mean;
    //vector<float> std;

    ifstream filein(argv[2]); 
    vector<string> imgs;
    for(string str; getline(filein, str); ) {
        vector<string> res = split_str(str, "\t");
        assert (!res.empty());
        imgs.push_back(res[0]);
    }
    float avr_time = 0, avr_preprocess_time = 0;
    Mat img_first = cv::imread(imgs[0], cv::IMREAD_COLOR);
    assert(!img_first.empty());
    Size input_size(640 ,640);
#ifdef USE_PADDING    
    Size output_size(img_first.cols, img_first.rows); //! cv::Size(cols, rows)
#else
    Size output_size(640, 640);
#endif     
    const int crop_start_x = input_size.width > output_size.width ? (input_size.width - output_size.width) / 2 : 0; // center aligned
    const int crop_start_y = input_size.height > output_size.height ? (input_size.height - output_size.height) / 2 : 0 ; 
    void * d_mean;
    void * d_std;
    void * d_out;
    cudaCheckError(cudaMalloc(&buffers[2], img_first.rows * img_first.cols * 3 * sizeof(uint8_t)));
    cudaCheckError(cudaMalloc(&d_mean, sizeof(float)*3 ));
    cudaCheckError(cudaMalloc(&d_std,  sizeof(float)*3 ));
    cudaCheckError(cudaMalloc(&d_out,  sizeof(uint8_t) * output_size.height * output_size.width));
    cudaCheckError(cudaMemcpy(d_mean, mean.data(), sizeof(float)*3, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_std,  std.data(),  sizeof(float)*3, cudaMemcpyHostToDevice));

    Mat img_raw;
    cout << "do inference ..." << endl;
    for(auto& img_name: imgs) {
        img_raw = cv::imread(img_name, cv::IMREAD_COLOR);
        assert(!img_raw.empty());

        auto startTime = std::chrono::high_resolution_clock::now();
#if USE_PADDING        
        preprocess_cuda(buffers[inputIndex], buffers[2], img_raw, input_size.height, input_size.width, ScaleOp::Padding,  
                d_mean, !mean.empty(), d_std,  !std.empty(), stream);
#else
        preprocess_cuda(buffers[inputIndex], buffers[2], img_raw, input_size.height, input_size.width, ScaleOp::Resize,  
                d_mean, !mean.empty(), d_std,  !std.empty(), stream);
#endif         
        context->enqueue(batchSize, buffers, stream, nullptr);
#if USE_PADDING        
        Mat out = postprocess_cuda(buffers[outputIndex],  input_size.height, input_size.width,  \
                         d_out, output_size.height, output_size.width, ScaleOp::Crop,  \
                         crop_start_x, crop_start_y,  stream, 2, 255);
#else 
        Mat out = postprocess_cuda(buffers[outputIndex],  input_size.height, input_size.width,  \
                         d_out, output_size.height, output_size.width, ScaleOp::Same,  \
                         crop_start_x, crop_start_y,  stream, 2, 255);
#endif        

        cudaStreamSynchronize(stream);
        auto endTime = std::chrono::high_resolution_clock::now();
        float infer_time = std::chrono::duration<float, std::milli>
                        (endTime - startTime).count();


        avr_time += infer_time;
        vector<string> tmp = split_str(img_name, "/");

        cout << "Image name:" << *tmp.rbegin() 
              //<< " preprocess cost:" << preprocess_time << " ms"
              << " Infer cost:" << infer_time << " ms" << endl;

        string out_name = "out/";
        out_name += *tmp.rbegin();
        cv::imwrite(out_name, out);
    }
    avr_time /= imgs.size(); 
    avr_preprocess_time /= imgs.size();
    cout << "Average Inference time: " << avr_time << " ms, " 
         << " preprocess cost:"  <<  avr_preprocess_time << " ms!"<< endl;

    for (auto& e: buffers) {
        cudaCheckError(cudaFree(e));
    }
    cudaCheckError(cudaFree(d_mean));
    cudaCheckError(cudaFree(d_std));
    cudaCheckError(cudaFree(d_out));
    cudaStreamDestroy(stream);




    return 0;
}

#else 
int main(int argc, char* argv[]) {
    const int gpu_id = 0;
    cudaCheckError(cudaSetDevice(gpu_id));
    int run_v = 0;
    int driver_v = 0;
    cudaCheckError(cudaRuntimeGetVersion(&run_v));
    cudaCheckError(cudaDriverGetVersion(&driver_v));
    cout << "cudaRuntimeVersion:" << run_v 
          << " cudaDriverGetVersion:" << driver_v 
            << endl;

    cudaStream_t stream; 
    cudaCheckError(cudaStreamCreate(&stream));    

    /// read serialized engine from local file 
    vector<char> trtModelStream_;
    size_t size{0};
    string file_name(argv[1]);
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
    cout << "size: " << size << endl;
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
        return -1;
    }



    /// allocate memory for input&output 
    const int batchSize = 1;

    int inputIndex = getBindingInputIndex(context.get());
    int outputIndex = 1 - inputIndex; 
    cout << "inputIdx:" << inputIndex << endl;
    void* buffers[2];

    vector<float> inputTensor;
    vector<float> outputTensor;
    Dims inputDims{4}; 
    Dims outputDims{4};
    for(int i = 0; i < engine->getNbBindings(); ++i) {
        auto dims = context ? context->getBindingDimensions(i):engine->getBindingDimensions(i);
        size_t vol = context ? 1 : static_cast<size_t>(batchSize);
        nvinfer1::DataType type = engine->getBindingDataType(i);
        int vecDim = engine->getBindingVectorizedDim(i);
        if(-1 != vecDim) {
            int scalarsPerVec = engine->getBindingComponentsPerElement(i);
            dims.d[vecDim] = divUp(dims.d[vecDim], scalarsPerVec);
            vol *= scalarsPerVec;
        }
        vol *= std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
        if(i == inputIndex) {
            inputTensor.resize(vol);
            inputDims = dims; 
            cout << "inputs' NCHW:" << dims.d[0] 
                << ", " << dims.d[1] << "," << dims.d[2]
                <<"," << dims.d[3] 
                <<" dataType:" << sizeof(type) << endl;
        } else {
            outputTensor.resize(vol);
            outputDims = dims;
            cout << "outputs' NCHW:" << dims.d[0] 
                << ", " << dims.d[1] << "," << dims.d[2]
                <<"," << dims.d[3] 
                <<" dataType:" << sizeof(type) << endl;
        }
        // cuda mem.
        cudaMalloc(&buffers[i], vol * sizeof(type));
    }
    /// mem. host->dev.
    /// dummy input
    int N = inputDims.d[0], C = inputDims.d[1], H = inputDims.d[2], W = inputDims.d[3];
    std::cout << " NCHW:" << N << ", " << C << "," << H <<"," << W << endl;
    at::Tensor dummy_input = at::randn({N, C, H, W}).clamp_min_(0) * 10 + 1.;
    auto ptr = static_cast<const float*>(dummy_input.data_ptr());
    memcpy(inputTensor.data(), dummy_input.data_ptr(), dummy_input.numel()* sizeof(float));

    cudaMemcpyAsync(buffers[inputIndex], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    //cudaMemcpy(buffers[inputIndex], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice);
    cout << "do inference ..." << endl;
    auto startTime = std::chrono::high_resolution_clock::now();
    context->enqueue(batchSize, buffers, stream, nullptr);
    cudaStreamSynchronize(stream);
    auto endTime = std::chrono::high_resolution_clock::now();
    float totalTime = std::chrono::duration<float, std::milli>
                        (endTime - startTime).count();
    cout << " Infer cost:" << totalTime << " ms" << endl;
    return 0;
    //cudaMemcpyAsync(outputTensor.data(), buffers[outputIndex], outputTensor.size()*sizeof(float), cudaMemcpyDeviceToHost, stream);
    //cudaStreamSynchronize(stream);
    cudaMemcpy(outputTensor.data(), buffers[outputIndex], outputTensor.size()*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 2; ++i) {
        cudaCheckError(cudaFree(buffers[i]));
    }
    cudaCheckError(cudaFree(d_mean));
    cudaCheckError(cudaFree(d_std));

    int output_h = outputDims.d[2];
    int output_w = outputDims.d[3];
    bool align_corners = true;
    at::Tensor out_tensor2 = at::upsample_bilinear2d(dummy_input, {output_h, output_w}, align_corners);

    auto ptr1 = static_cast<const float*>(outputTensor.data());
    auto ptr2 = static_cast<const float*>(out_tensor2.data_ptr());
    // verify
    for(size_t i = 0; i < out_tensor2.numel(); ++i) {
       float err = fabs(ptr1[i] - ptr2[i]);
       assert(err < 1e-5);
    }




    return 0;
}
#endif 
