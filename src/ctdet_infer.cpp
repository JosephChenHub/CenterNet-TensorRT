#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <NvInfer.h>
#include <cassert>

#include <cuda_profiler_api.h> 

#include "common/logger.h"
#include "dcn_v2.hpp" //! DCN plugin

#include "gpu_sort.hpp"
#include "det_kernels.hpp"
#include "custom.hpp"

using namespace std;
using namespace cv;
using namespace nvinfer1;

#define CHECK_CUDA(e) { if(e != cudaSuccess) { \
    printf("cuda failure: %s:%d: '%s'\n", __FILE__, __LINE__, \
            cudaGetErrorString(e)); \
        exit(0); \
    } \
}

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

const vector<string> coco_class_name {
     "person", "bicycle", "car", "motorcycle", "airplane",
     "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
     "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
     "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
     "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
     "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
     "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
     "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
     "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
     "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
     "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
     "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
     "scissors", "teddy bear", "hair drier", "toothbrush"};

vector<cv::Scalar> colors(80);




int main(int argc, char* argv[]) {
    CHECK_CUDA(cudaSetDevice(0));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    /// read the serialized engine 
    string trt_file(argv[1]); 
    vector<char> trtModelStream_;
    size_t size(0);
    cout << "Loading engine file:" << trt_file << endl;
    ifstream file(trt_file, std::ios::binary);
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
    auto engine = unique_ptr<ICudaEngine, NvInferDeleter>(runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr));
    if(!engine) {
        cerr << " Failed to create the engine from .trt file!" << endl;
    } else {
        cout << " Create the engine from " << trt_file << " successfully!" << endl;
    }
    /// an execution context holds additional memory to store intermediate activation values. an engine can have multiple contexts sharing the same weights for multi-tasks/streams
    auto context = unique_ptr<IExecutionContext, NvInferDeleter>(engine->createExecutionContext());
    if(!context) {
        cerr << " Failed to createExecutionContext!" << endl;
        exit(-1);
    }
    const int nb_bindings = context->getEngine().getNbBindings();
    map<string, pair<int, size_t>> engine_name_size;
    vector<string> binding_names(nb_bindings);
    string input_name = "input";
    for(int i = 0; i < nb_bindings; ++i) {
        auto dim = context->getEngine().getBindingDimensions(i);
        string name = context->getEngine().getBindingName(i);
        size_t size = dim.d[0] * dim.d[1] * dim.d[2] * dim.d[3];
        size_t pos = name.find("input");
        if (pos != name.npos) input_name = name;
        cout << "i=" << i  << " tensor's name:"
            << name
            << " dim:("
            << dim.d[0] << ","
            << dim.d[1] << ","
            << dim.d[2] << ","
            << dim.d[3] << ")" 
            << " size:" << size
            << endl;
        binding_names[i] = name;
        engine_name_size.emplace(name, make_pair(i, size));
    }
    int input_idx = context->getEngine().getBindingIndex(input_name.c_str());
    int hm_idx = context->getEngine().getBindingIndex("hm");
    int wh_idx = context->getEngine().getBindingIndex("wh");
    int reg_idx = context->getEngine().getBindingIndex("reg");
    cout <<"buffers'index, input:" << input_idx << " hm:" << hm_idx << " wh:"
        << wh_idx << " reg:" << reg_idx << endl;

    /// memory allocation
    string img_name(argv[2]);
    Mat img = imread(img_name);
    assert (img.data != NULL);

    const int batch_num = 1;
    const int num_classes = 80;
    const int K = 100;
    const int net_h = 512;
    const int net_w = 512;
    const int down_ratio = 4;
    const int net_oh = 128;
    const int net_ow = 128;
    const size_t input_size = img.rows * img.cols * 3 * batch_num; //! directly copy cv::Mat to GPU mem.
    const size_t hm_size = net_oh * net_ow * num_classes * batch_num;
    const size_t wh_size = net_oh * net_ow * 2 * batch_num;
    const int det_len = 1 + K * 6;
    const size_t det_size = det_len * batch_num;

    uint8_t* d_in;
    float* buffers[4];
    size_t* d_indices;
    float* d_det;
    float* h_det;

    cv::RNG rng(time(0));
    for (int  i = 0; i < 80; ++i) {
        colors[i] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }
    /// calculate the affine transform matrix
    float* d_inv_trans;
    float h_inv_trans[12];
    CHECK_CUDA(cudaMalloc((void**)&d_inv_trans, sizeof(float)*12));
    float center[2], scale[2];
    center[0] = img.cols / 2.;
    center[1] = img.rows / 2.;
    scale[0] = img.rows > img.cols ? img.rows : img.cols;
    scale[1] = scale[0];

    float shift[2] = {0., 0.};
    get_affine_transform(h_inv_trans, center, scale, shift, 0, net_h, net_w, true); //! src -> dst
    get_affine_transform(h_inv_trans+6, center, scale, shift, 0, net_h / down_ratio, net_w / down_ratio, true); // det's

    CHECK_CUDA(cudaMemcpyAsync(d_inv_trans, h_inv_trans, sizeof(float) * 12, cudaMemcpyHostToDevice, stream));
    /// 
    CHECK_CUDA(cudaMallocHost((void**)&h_det, sizeof(float) * det_size));
    CHECK_CUDA(cudaMalloc((void**)&d_det, sizeof(float) * det_size));
    CHECK_CUDA(cudaMalloc((void**)&d_in, sizeof(uint8_t) * input_size));
    CHECK_CUDA(cudaMalloc((void**)&buffers[input_idx], sizeof(float) * input_size));
    CHECK_CUDA(cudaMalloc((void**)&buffers[hm_idx], sizeof(float) * hm_size));
    CHECK_CUDA(cudaMalloc((void**)&buffers[wh_idx], sizeof(float) * wh_size));
    CHECK_CUDA(cudaMalloc((void**)&buffers[reg_idx], sizeof(float) * wh_size));
    CHECK_CUDA(cudaMalloc((void**)&d_indices, sizeof(size_t) * hm_size));
    CHECK_CUDA(cudaMemsetAsync(d_det, 0, sizeof(float) * det_size, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_in, img.data, sizeof(uint8_t) * input_size, cudaMemcpyHostToDevice, stream));
    //CHECK_CUDA(cudaProfilerStart());


    float* d_mean;
    float* d_std;
    //vector<float> mean {0.408, 0.447, 0.470}; // BGR mode
    //vector<float> std {0.289, 0.274, 0.278};
    //CHECK_CUDA(cudaMalloc((void**)&d_mean, sizeof(float)*3));
    //CHECK_CUDA(cudaMalloc((void**)&d_std, sizeof(float)*3));
    //CHECK_CUDA(cudaMemcpyAsync(d_mean, mean.data(), sizeof(float)*3, cudaMemcpyHostToDevice, stream));
    //CHECK_CUDA(cudaMemcpyAsync(d_std, std.data(), sizeof(float)*3, cudaMemcpyHostToDevice, stream));

    cuda_centernet_preprocess(batch_num, 
        d_in, 3,  img.rows, img.cols,   
        buffers[input_idx], net_h, net_w, 
        d_inv_trans, d_mean, true, 
        d_std, true, stream);

    cout << " starting inference ... " << endl;
    context->enqueue(batch_num, (void**)buffers, stream, nullptr);

    /// decode the detection's result
    ctdet_decode(d_det, buffers[wh_idx], buffers[reg_idx], 
            buffers[hm_idx], d_indices, 
            d_inv_trans+6,  
            batch_num, num_classes, net_oh, net_ow, K, 
            0.3, true, false, stream);
    CHECK_CUDA(cudaMemcpyAsync(h_det, d_det, sizeof(float) * det_size, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    //cudaDeviceSynchronize();

    int batch_id = 0;
    int num_bbox = static_cast<int>(h_det[batch_id * det_len]);
    if (!num_bbox) {
        cout << " No objects detected in the image!" << endl;
    } else {
        cout << num_bbox << " objects have been detected in the image!" << endl;
        const int font_face = FONT_HERSHEY_SIMPLEX;
        const double font_scale = 0.5;
        const int thickness = 1.5;
        for (int i = 0; i < num_bbox; ++i) {
            /// visulize
            int x0 = h_det[batch_id * det_len + 1 + i * 6 + 2];
            int y0 = h_det[batch_id * det_len + 1 + i * 6 + 3]; 
            int x1 = h_det[batch_id * det_len + 1 + i * 6 + 4];
            int y1 = h_det[batch_id * det_len + 1 + i * 6 + 5];
            int cls_id = h_det[batch_id * det_len + 1 + i * 6 + 0];
            float score = h_det[batch_id * det_len + 1 + i * 6 + 1];
            cout << "class:" << coco_class_name[cls_id] << " score:"
                << score << " bbox:[" << x0 << "," 
                << y0  << "," << x1 << "," << y1 << "]\n";

            string text = coco_class_name[cls_id];
            text += ":";
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << score;
            text += ss.str();

            int baseline=0;
            cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

            cv::rectangle(img, cv::Point(x0, y0), cv::Point(x1, y1), colors[cls_id], 3);
            //cv::rectangle(img, cv::Point(x0, y0 - text_size.height - 2), \
                    cv::Point(x0 + text_size.width, y0 - 2), colors[cls_id], 2);
            cv::putText(img, text, cv::Point(x0, y0-5), font_face, font_scale, cv::Scalar(0, 0, 255), thickness, cv::LINE_AA);

        }
        cout << endl;

        string out_name = "det_out/det_";

        size_t index = img_name.find_last_of("/\\");
        if (index != img_name.npos)  {
            out_name += img_name.substr(index+1);
        } else {
            out_name += img_name;
        }
        cout << "Saving the image to " << out_name << endl;

        cv::imwrite(out_name, img); 
    }



    //CHECK_CUDA(cudaProfilerStop());
    //CHECK_CUDA(cudaFree(d_mean));
    //CHECK_CUDA(cudaFree(d_std));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_indices));
    CHECK_CUDA(cudaFree(d_det));
    CHECK_CUDA(cudaFree(d_inv_trans));
    CHECK_CUDA(cudaFreeHost(h_det));
    for(int i = 0;i < 4; ++i) CHECK_CUDA(cudaFree(buffers[i]));
    CHECK_CUDA(cudaStreamDestroy(stream));


    return 0;
}
