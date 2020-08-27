#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <utility>
#include <map>
#include <fstream>
#include <memory>
#include <NvInfer.h>
#include <cassert>

#include "common/logger.h"
#include "dcn_v2.hpp" //! DCN plugin

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



const vector<pair<int, int>> edges {
    make_pair(0, 1), make_pair(0, 2), make_pair(1, 3), make_pair(2, 4),
    make_pair(3, 5), make_pair(4, 6), make_pair(5, 6), make_pair(5, 7),
    make_pair(7, 9), make_pair(6, 8), make_pair(8, 10),make_pair(5, 11),
    make_pair(6, 12), make_pair(11, 12), make_pair(11, 13), make_pair(13, 15),
    make_pair(12, 14), make_pair(14, 16)};

const vector<Scalar> e_colors {
    Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(255, 0, 0), Scalar(0, 0, 255), 
    Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(255, 0, 255), Scalar(255, 0, 0), 
    Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(0, 0, 255), Scalar(255, 0, 0), 
    Scalar(0, 0, 255), Scalar(255, 0, 255), Scalar(255, 0, 0), Scalar(255, 0, 0), 
    Scalar(0, 0, 255), Scalar(0, 0, 255)};

const vector<Scalar> hp_colors {
    Scalar(255, 0, 255), Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(255, 0, 0), 
    Scalar(0, 0, 255), Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(255, 0, 0), 
    Scalar(0, 0, 255), Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(255, 0, 0), 
    Scalar(0, 0, 255), Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(255, 0, 0),
    Scalar(0, 0, 255) };


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
            cerr << "FATAL: dynamic shape is not supported currently!" << endl;
            return -1;
        }
        size_t size = dim.d[0] * dim.d[1] * dim.d[2] * dim.d[3];
        size_t pos = name.find("input");
        if (pos != name.npos) {
            if (dim.d[1] != 3) {
                cerr << "FATAL: channel must be 3!" << endl;
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
    // 
    cv::RNG rng(time(0));
    vector<Scalar> colors(80);
    for (int  i = 0; i < 80; ++i) {
        colors[i] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }

    // allocate memory 
    const int batch_size = runtime_input_dims[0];
    const int channel = runtime_input_dims[1];
    const int net_h = runtime_input_dims[2];
    const int net_w = runtime_input_dims[3];
    const int down_ratio = 4;
    const int net_oh = net_h / down_ratio;
    const int net_ow = net_w / down_ratio;
    const size_t raw_img_size = img_h * img_w * channel;
    const int K = 100;
    const int num_joints = 17;
    const int det_len = 2 + 4 + num_joints*2; // score + class+points
    const int batch_len = 1 + det_len * K; 
    const size_t det_size = batch_len * batch_size;


    unique_ptr<GPUAllocator> gpu_mem(new GPUAllocator);

    float* buffers[32];
    for(int i = 0; i < nb_bindings; ++i) {
        buffers[i] = gpu_mem->allocate<float>(binding_names[i].c_str(),  engine_name_size[binding_names[i]].second );
    }
    auto d_inv_trans = gpu_mem->allocate<float>("inv_trans", 12);
    auto d_det = gpu_mem->allocate<float>("det", det_size);
    auto d_imgs = gpu_mem->allocate<uint8_t>("raw_imgs", raw_img_size*batch_size);
    auto d_heat_ind = gpu_mem->allocate<size_t>("heat", engine_name_size["hm"].second);
    auto d_hp_ind = gpu_mem->allocate<size_t>("hp", engine_name_size["hm_hp"].second);
    CHECK_CUDA(cudaMemset(d_det, 0, sizeof(float) * det_size));

    float* h_det;
    CHECK_CUDA(cudaMallocHost((void**)&h_det, sizeof(float) * det_size));


    // calculate the affine transformation matrix
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


    cout << " starting inference ... " << endl;
    vector<cv::Mat> imgs(batch_size);
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
            buffers[engine_name_size[input_name].first], net_h, net_w, 
            d_inv_trans, nullptr, true, 
            nullptr, true, stream);
        // forward 
        context->enqueue(batch_size, (void**)buffers, stream, nullptr);

        // decode the detection's result
        multi_pose_decode(d_det, 
                buffers[engine_name_size["hm"].first],
                buffers[engine_name_size["wh"].first],
                buffers[engine_name_size["reg"].first], 
                buffers[engine_name_size["hps"].first],
                buffers[engine_name_size["hm_hp"].first], 
                buffers[engine_name_size["hp_offset"].first], 
                d_heat_ind, d_hp_ind, d_inv_trans+6,  
                batch_size, num_joints, net_oh, net_ow, K, 
                0.3, true, true, stream);
        CHECK_CUDA(cudaMemcpyAsync(h_det, d_det, sizeof(float) * det_size, cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // visualization 
        for (int batch_id = 0; batch_id < local_id; ++batch_id) {
            int num_bbox = static_cast<int>(h_det[batch_id * batch_len]);
            auto img = imgs[batch_id];
            if (!num_bbox) {
                cout << " No human detected in the image!" << endl;
                continue; 
            } else {
                cout << num_bbox << " human detected in the image!" << endl;
                const int font_face = FONT_HERSHEY_SIMPLEX;
                const double font_scale = 0.5;
                const int thickness = 1.5;

                for (int i = 0; i < num_bbox; ++i) {
                    /// visulize
                    int x0 = h_det[batch_id * batch_len + 1 + i * det_len + 2];
                    int y0 = h_det[batch_id * batch_len + 1 + i * det_len + 3]; 
                    int x1 = h_det[batch_id * batch_len + 1 +  i * det_len + 4];
                    int y1 = h_det[batch_id * batch_len + 1 + i * det_len + 5];
                    int cls_id = h_det[batch_id * batch_len + 1 + i * det_len + 0];
                    float score = h_det[batch_id * batch_len + 1 + i * det_len + 1];
                    cout << " score:"
                        << score << " bbox:[" << x0 << "," 
                        << y0  << "," << x1 << "," << y1 << "]\n";

                    string text = "person";
                            text += ":";
                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(2) << score;
                    text += ss.str();

                    int baseline=0;
                    cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

                    cv::rectangle(img, cv::Point(x0, y0), cv::Point(x1, y1), colors[cls_id], 5);
                    cv::putText(img, text, cv::Point(x0, y0-3), font_face, font_scale, cv::Scalar(0, 0, 255), thickness, cv::LINE_AA);
                    for(int j = 0; j < num_joints; ++j) {
                        x0 = h_det[batch_id * batch_len + 1 + i * det_len + 6 + j*2];
                        y0 = h_det[batch_id * batch_len + 1 + i * det_len + 6 + j*2 + 1];
                        cv::circle(img, cv::Point(x0, y0), 6, hp_colors[i], -1);
                    }
                    for(int j = 0; j < edges.size(); ++j) {
                        int p0 = edges[j].first;
                        int p1 = edges[j].second;

                        x0 = h_det[batch_id*batch_len + 1 + i*det_len + 6 + p0*2];
                        y0 = h_det[batch_id*batch_len + 1 + i*det_len + 6 + p0*2+1];
                        x1 = h_det[batch_id*batch_len + 1 + i*det_len + 6 + p1*2];
                        y1 = h_det[batch_id*batch_len + 1 + i*det_len + 6 + p1*2+1];
                        if (x0 > 0 && y0 > 0 && x1 > 0 && y1 > 0) {
                            cv::line(img, cv::Point(x0, y0), cv::Point(x1, y1), e_colors[j], 4, cv::LINE_AA);
                        }
                    }
                }
                cout << endl;

                string out_name = output_path + "/pos_";

                auto img_name = input_imgs[idx+batch_id];
                size_t index = img_name.find_last_of("/\\");
                if (index != img_name.npos)  {
                    out_name += img_name.substr(index+1);
                } else {
                    out_name += img_name;
                }
                cout << "Saving the image to " << out_name << endl;
                cv::imwrite(out_name, img); 
            }
        }
        idx += local_id;
    }


    //CHECK_CUDA(cudaProfilerStop());
    CHECK_CUDA(cudaFreeHost(h_det));
    CHECK_CUDA(cudaStreamDestroy(stream));


    return 0;
}

