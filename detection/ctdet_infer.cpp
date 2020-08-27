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

using namespace std;
using namespace cv;
using namespace nvinfer1;


struct NvInferDeleter {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};


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

    // set device 
    CHECK_CUDA(cudaSetDevice(device_id));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // read the serialized engine 
    string trt_file(engine_file); 
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
    } else {
        cerr << "Failed to open the engine file:" << trt_file << endl;
        return -1;
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
    // an execution context holds additional memory to store intermediate activation values. an engine can have multiple contexts sharing the same weights for multi-tasks/streams
    auto context = unique_ptr<IExecutionContext, NvInferDeleter>(engine->createExecutionContext());
    if(!context) {
        cerr << " Failed to createExecutionContext!" << endl;
        return -1;
    }
    // parse the dimensions of inputs and outputs. 
    const int nb_bindings = context->getEngine().getNbBindings();
    map<string, pair<int, size_t>> engine_name_size;
    vector<string> binding_names(nb_bindings);
    string input_name = "input";
    vector<int> runtime_input_dims {0, 0, 0, 0};
    for(int i = 0; i < nb_bindings; ++i) {
        auto dim = context->getEngine().getBindingDimensions(i);
        string name = context->getEngine().getBindingName(i);
        if (dim.d[0] == -1 || dim.d[1] == -1 || dim.d[2] == -1 || dim.d[3] == -1) {
            cerr << "Fatal: dynamic shape is not supported currently!" << endl;
            return -1;
        }
        size_t size = dim.d[0] * dim.d[1] * dim.d[2] * dim.d[3];
        size_t pos = name.find("input");
        if (pos != name.npos) {
            if (dim.d[1] != 3) {
                 cerr << "Fatal: input channel must be 3, where the dim is" << dim.d[1] << endl;
                 return -1;
            }
            input_name = name;
            runtime_input_dims[0] = dim.d[0];
            runtime_input_dims[1] = dim.d[1];
            runtime_input_dims[2] = dim.d[2];
            runtime_input_dims[3] = dim.d[3];
        }
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

    // for visualization 
    cv::RNG rng(time(0));
    vector<cv::Scalar> colors(80);
    for (int  i = 0; i < 80; ++i) {
        colors[i] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }

    // memory allocation 
    const int num_classes = 80;
    const int K = 100;
    const int batch_size = runtime_input_dims[0];
    const int channel = runtime_input_dims[1]; // default 3
    const int net_h = runtime_input_dims[2]; // default 512
    const int net_w = runtime_input_dims[3]; // default 512
    const int down_ratio = 4;
    const int net_oh = net_h / down_ratio;
    const int net_ow = net_w / down_ratio;


    const size_t input_size = net_h * net_w * channel * batch_size; 
    const size_t hm_size = net_oh * net_ow * num_classes * batch_size;
    const size_t wh_size = net_oh * net_ow * 2 * batch_size;
    const int det_len = 1 + K * 6;
    const size_t det_size = det_len * batch_size;
    const size_t raw_img_size = img_h * img_w * channel;  


    unique_ptr<GPUAllocator> gpu_mem(new GPUAllocator);

    float* buffers[4];
    auto d_imgs = gpu_mem->allocate<uint8_t>("raw_imgs", raw_img_size * batch_size);
    buffers[input_idx] = gpu_mem->allocate<float>("input", input_size);
    buffers[hm_idx] = gpu_mem->allocate<float>("hm", hm_size);
    buffers[wh_idx] = gpu_mem->allocate<float>("wh", wh_size);
    buffers[reg_idx] = gpu_mem->allocate<float>("reg", wh_size);

    auto d_inv_trans = gpu_mem->allocate<float>("inv_trans", 12);

    auto d_det = gpu_mem->allocate<float>("det", det_size);
    auto d_indices = gpu_mem->allocate<size_t>("indices", hm_size);



    // calculate the affine transform matrix
    float h_inv_trans[12];
    float center[2], scale[2];
    center[0] = img_w / 2.;
    center[1] = img_h / 2.;
    scale[0] = img_h > img_w ? img_h : img_w;
    scale[1] = scale[0];

    float shift[2] = {0., 0.};
    get_affine_transform(h_inv_trans, center, scale, shift, 0, net_h, net_w, true); //! src -> dst
    get_affine_transform(h_inv_trans+6, center, scale, shift, 0, net_h / down_ratio, net_w / down_ratio, true); // det's
    CHECK_CUDA(cudaMemcpyAsync(d_inv_trans, h_inv_trans, sizeof(float) * 12, cudaMemcpyHostToDevice, stream));

    float * h_det;
    CHECK_CUDA(cudaMallocHost((void**)&h_det, sizeof(float) * det_size));
    CHECK_CUDA(cudaMemsetAsync(d_det, 0, sizeof(float) * det_size, stream));

    
    vector<cv::Mat> imgs (batch_size); 

    cout << " starting inference ... " << endl;
    //CHECK_CUDA(cudaProfilerStart());
    for(size_t idx = 0; idx < input_imgs.size();  ) {
        // read raw images to gpu
        int local_id = 0;
        for( ; local_id < batch_size && (idx+local_id) < input_imgs.size(); ) {
            Mat raw_img = cv::imread(input_imgs[idx+local_id]);
            imgs[local_id] = raw_img;
            CHECK_CUDA(cudaMemcpy(d_imgs + raw_img_size * local_id, raw_img.data, sizeof(uint8_t) * raw_img_size, cudaMemcpyHostToDevice));
            ++local_id; 
        }
        // pre-processing 
        cuda_centernet_preprocess(d_imgs, 
            batch_size, channel,  img_h, img_w, 
            buffers[input_idx], net_h, net_w, 
            d_inv_trans, nullptr, true, 
            nullptr, true, stream);

        // forward 
        context->enqueue(batch_size, (void**)buffers, stream, nullptr);

        // decode 
        ctdet_decode(d_det, buffers[wh_idx], buffers[reg_idx], 
                buffers[hm_idx], d_indices, 
                d_inv_trans+6,  
                batch_size, num_classes, net_oh, net_ow, K, 
                0.3, true, false, stream);
        CHECK_CUDA(cudaMemcpyAsync(h_det, d_det, sizeof(float) * det_size, cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // visualization 
        for (int batch_id = 0; batch_id < local_id; ++batch_id) {
            int num_bbox = static_cast<int>(h_det[batch_id * det_len]);
            if (!num_bbox) {
                cout << " No objects detected in the image!" << endl;
                continue; 
            } else {
                cout << num_bbox << " objects have been detected in the image!" << endl;
                const int font_face = FONT_HERSHEY_SIMPLEX;
                const double font_scale = 0.5;
                const int thickness = 1.5;
                for (int i = 0; i < num_bbox; ++i) {
                    // visulize
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

                    cv::rectangle(imgs[batch_id], cv::Point(x0, y0), cv::Point(x1, y1), colors[cls_id], 3);
                    cv::putText(imgs[batch_id], text, cv::Point(x0, y0-5), font_face, font_scale, cv::Scalar(0, 0, 255), thickness, cv::LINE_AA);
                }
                cout << endl;

                string out_name = output_path + "/det_";

                string img_name = input_imgs[idx+batch_id];
                size_t index = img_name.find_last_of("/\\");
                if (index != img_name.npos)  {
                    out_name += img_name.substr(index+1);
                } else {
                    out_name += img_name;
                }
                cout << "Saving the image to " << out_name << endl;
                cv::imwrite(out_name, imgs[batch_id]); 
            }
        }
        // 
        idx += local_id;
    }

    //CHECK_CUDA(cudaProfilerStop());
    CHECK_CUDA(cudaFreeHost(h_det));
    CHECK_CUDA(cudaStreamDestroy(stream));


    return 0;
}
