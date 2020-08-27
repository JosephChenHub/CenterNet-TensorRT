#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <NvInfer.h>
#include <cassert>

#include <cuda_profiler_api.h> 

#include "common/logger.h"
#include "dcn_v2.hpp" // DCN plugin

#include "gpu_common.cuh"
#include "custom.hpp"
#include "process.hpp" 
#include "decode.hpp"

#include "infer.hpp"

using namespace std;
using namespace cv;
using namespace nvinfer1;




vector<string> coco_class_name {
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



int main(int argc, char* argv[]) {
    // parse the arguments
    const char* cmd_keys = "{help h usage? | |print this message}"
                           "{g | 0 | device id of gpus}"
                           "{img_h | 512 | input images' height}"
                           "{img_w | 512 | input images' width}"
                           "{e | engine.trt | the serialized engine file}"
                           "{i | data.txt | text file of the input images}"
                           "{o | result | path to the output files}";
                            

    cv::CommandLineParser parser(argc, argv, cmd_keys);
    parser.about("CenterNet deploy");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    const int device_id = parser.get<int>("g");
    const int img_h = parser.get<int>("img_h");
    const int img_w = parser.get<int>("img_w");
    const string engine_file = parser.get<string>("e");
    const string input_txt = parser.get<string>("i");
    string output_path = parser.get<string>("o");
    if (*output_path.rbegin() == '/') output_path.erase(output_path.end()-1, output_path.end());
    makedirs(output_path.c_str());

    vector<string> input_imgs;
    ifstream text_file(input_txt);
    assert(text_file.good());
    string line;
    while(getline(text_file, line)) {
        vector<string> tmp = split_str(line, " ");
        input_imgs.push_back(tmp[0]);
    }
    if (input_imgs.empty()) {
        cerr << "Fatal: please check the input text file:" << input_txt << endl;
        return -1;
    }
    for(auto &img_name: input_imgs) {
        auto data = cv::imread(img_name);
        if (data.empty() || data.rows != img_h || data.cols != img_w) {
            cerr << "Fatal: please check the input image:" << img_name << endl;
            return -1;
        }
    }
    // for visualization 
    cv::RNG rng(time(0));
    vector<cv::Scalar> colors(80);
    for (int  i = 0; i < 80; ++i) {
        colors[i] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }


    // trt inference
    const int num_classes = 80;
    using namespace TRT; 
    unique_ptr<TRTInferBase> trt_infer(new TRTCenterNetDet(engine_file.c_str(), device_id, "input", 
            img_h, img_w, num_classes));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    size_t batch_size = trt_infer->batchSize(); 
    vector<cv::Mat> imgs (batch_size); 
    
    auto d_imgs = trt_infer->data_pointer<uint8_t>();
    size_t raw_img_size = trt_infer->raw_image_size(); 
    void* vis_argv[32]; 

    cout << " starting inference ... " << endl;
    //CHECK_CUDA(cudaProfilerStart());
    for(size_t idx = 0; idx < input_imgs.size();  ) {
        // read raw images to gpu
        int local_id = 0;
        for( ; local_id < batch_size && (idx+local_id) < input_imgs.size(); ) {
            imgs[local_id] = cv::imread(input_imgs[idx + local_id]);
            CHECK_CUDA(cudaMemcpyAsync(d_imgs + raw_img_size * local_id, imgs[local_id].data, sizeof(uint8_t) * raw_img_size, cudaMemcpyHostToDevice, stream));
            ++local_id; 
        }

        trt_infer->infer(stream); 

        vis_argv[0] = reinterpret_cast<void*>(&input_imgs);
        vis_argv[1] = reinterpret_cast<void*>(&imgs);
        vis_argv[2] = reinterpret_cast<void*>(&coco_class_name);
        vis_argv[3] = reinterpret_cast<void*>(&colors);
        vis_argv[4] = reinterpret_cast<void*>(&output_path);
        vis_argv[5] = reinterpret_cast<void*>(&local_id);
        vis_argv[6] = reinterpret_cast<void*>(&idx);

        trt_infer->visualize(7, vis_argv); 

        // 
        idx += local_id;
    }

    //CHECK_CUDA(cudaProfilerStop());
    CHECK_CUDA(cudaStreamDestroy(stream));


    return 0;
}
