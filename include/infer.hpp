#pragma once

#include "base.hpp"
#include <opencv2/opencv.hpp>


namespace TRT {
   using namespace cv;
   using namespace std;
class TRTCenterNetDet final: public TRTInferBase {
private:
    int _img_h;
    int _img_w;
    int _channel; 
    int _batch_size;
    int _net_out_hw[2];

    int _num_classes; 

    int _wh_idx;
    int _hm_idx;
    int _reg_idx;
    const int _K = 100; 
    const int _det_len = 1 + _K * 6;
    const int _down_ratio = 4;
    float* _d_inv_trans;

    float _h_inv_trans[12];
    float * _h_det;
    float * _d_det;
    size_t* _d_indices;
    const float _threshold = 0.3; 
public:
    TRTCenterNetDet (const char* engine_name, const int device_id, const char* input_name, 
            const int img_h, const int img_w, const int num_classes) : TRTInferBase(engine_name, device_id, 
                false, input_name), _img_h(img_h), _img_w(img_w),
            _num_classes(num_classes) {
        _batch_size = _runtime_input_dims[0];
        _channel = _runtime_input_dims[1];

        _net_out_hw[0] = _runtime_input_dims[2] / _down_ratio; // TODO: how about ratio is float point?
        _net_out_hw[1] = _runtime_input_dims[3] / _down_ratio;

	    mem_alloc();

        // calculate the affine transform matrix
        float center[2], scale[2];
        center[0] = img_w / 2.;
        center[1] = img_h / 2.;
        scale[0] = img_h > img_w ? img_h : img_w;
        scale[1] = scale[0];

        float shift[2] = {0., 0.};
        get_affine_transform(_h_inv_trans, center, scale, shift, 0, _runtime_input_dims[2], _runtime_input_dims[3], true); //! src -> dst
        get_affine_transform(_h_inv_trans+6, center, scale, shift, 0, _net_out_hw[0], _net_out_hw[1], true); // det's
        CHECK_CUDA(cudaMemcpy(_d_inv_trans, _h_inv_trans, sizeof(float) * 12, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMallocHost((void**)&_h_det, sizeof(float) * _det_len * _batch_size));
        CHECK_CUDA(cudaMemset(_d_det, 0, sizeof(float) * _det_len * _batch_size));

	    _mem_initialized = true;
    }
    ~TRTCenterNetDet() {  
        CHECK_CUDA(cudaFreeHost(_h_det));
    }
protected:
    void pre_process(cudaStream_t stream) override {
        auto ptr = reinterpret_cast<uint8_t*>(_d_imgs);
        auto index = getBindingIndex(_input_name.c_str());
        cuda_centernet_preprocess(ptr, _runtime_input_dims[0], _runtime_input_dims[1],
                _img_h, _img_w, _buffers[index], 
                _runtime_input_dims[2], _runtime_input_dims[3], 
                _d_inv_trans, nullptr, true, nullptr, true, 
                stream);
    }

    void post_process(cudaStream_t stream) override {
        ctdet_decode(_d_det, _buffers[_wh_idx], _buffers[_reg_idx], 
                _buffers[_hm_idx], _d_indices, 
                _d_inv_trans+6,  
                _batch_size, _num_classes, _net_out_hw[0], _net_out_hw[1], _K, 
                _threshold, true, false, stream);
        CHECK_CUDA(cudaMemcpyAsync(_h_det, _d_det, sizeof(float) * _det_len * _batch_size, cudaMemcpyDeviceToHost, stream));
    }
    void mem_alloc() override {
        size_t raw_img_size = _img_h * _img_w * _channel; 
        size_t input_size = _runtime_input_dims[0] * _runtime_input_dims[1] * _runtime_input_dims[2] * _runtime_input_dims[3];
        size_t hm_size = _net_out_hw[0] * _net_out_hw[1] * _num_classes * _batch_size;
        size_t wh_size = _net_out_hw[0] * _net_out_hw[1] * 2 * _batch_size;
        size_t det_size = _det_len * _batch_size;
         
        gLogInfo << "allocation contents -- batch_size:" << _batch_size
		<< " raw_img_size:" << raw_img_size
		<< " input_size:" << input_size
		<< " hm_size:" << hm_size
		<< " wh_size:" << wh_size
		<<" reg_size:" << wh_size
		<< " det_size :" << det_size
		<< endl;

        uint8_t* raw_data = _gpu_mem->allocate<uint8_t>("raw_imgs", raw_img_size * _batch_size);
        _d_imgs = reinterpret_cast<void*>(raw_data);


        int input_idx = getBindingIndex(_input_name.c_str()); 
        _buffers[input_idx] =  _gpu_mem->allocate<float>("net_in", input_size);

        _hm_idx = getBindingIndex("hm");
        _buffers[_hm_idx] = _gpu_mem->allocate<float>("hm", hm_size);
        _wh_idx = getBindingIndex("wh");
        _buffers[_wh_idx] = _gpu_mem->allocate<float>("wh", wh_size);
        _reg_idx = getBindingIndex("reg");
        _buffers[_reg_idx] = _gpu_mem->allocate<float>("reg", wh_size);


        _d_inv_trans = _gpu_mem->allocate<float>("inv_trans", 12);
        _d_det = _gpu_mem->allocate<float>("det", det_size);
        _d_indices = _gpu_mem->allocate<size_t>("indices", hm_size);
    }

    size_t raw_image_size() const override {
        return _img_h * _img_w * _channel; 
    }
public:
    void visualize(int argc, void* argv[]) override {
        if (argc != 7) {
            gLogFatal << "Please check the number of the input var.!" << std::endl;
            exit(-1);
        }
    
        auto img_names = reinterpret_cast<std::vector<std::string>*>(argv[0]); 
        auto imgs = reinterpret_cast<std::vector<cv::Mat>*>(argv[1]);
        auto class_names = reinterpret_cast<std::vector<std::string>*>(argv[2]);
	    auto colors = reinterpret_cast<std::vector<cv::Scalar>*>(argv[3]);
        auto saved_dir = reinterpret_cast<std::string*>(argv[4]);
        auto batch_size = reinterpret_cast<int*>(argv[5]);
        auto start_id = reinterpret_cast<int*>(argv[6]);

        this->visualize(*img_names, *imgs, *class_names, *colors, *saved_dir, *batch_size, *start_id);

    }

public:
    void visualize(std::vector<std::string>& img_names, std::vector<cv::Mat>& imgs, 
            std::vector<std::string>& class_names, std::vector<cv::Scalar>& colors, 
            std::string& saved_dir, 
            const int batch_size, const int start_id) {
        for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
            int num_bbox = static_cast<int>(_h_det[batch_id * _det_len]);
            if (!num_bbox) {
                gLogInfo << " No objects detected in the image!" << std::endl;
                continue; 
            } else {
                gLogInfo << num_bbox << " objects have been detected in the image!" << std::endl;
                const int font_face = FONT_HERSHEY_SIMPLEX;
                const double font_scale = 0.5;
                const int thickness = 1.5;
                for (int i = 0; i < num_bbox; ++i) {
                    // visulize
                    int x0 = _h_det[batch_id * _det_len + 1 + i * 6 + 2];
                    int y0 = _h_det[batch_id * _det_len + 1 + i * 6 + 3]; 
                    int x1 = _h_det[batch_id * _det_len + 1 + i * 6 + 4];
                    int y1 = _h_det[batch_id * _det_len + 1 + i * 6 + 5];
                    int cls_id = _h_det[batch_id * _det_len + 1 + i * 6 + 0];
                    float score = _h_det[batch_id * _det_len + 1 + i * 6 + 1];
                    gLogInfo << "class:" << class_names[cls_id] << " score:"
                        << score << " bbox:[" << x0 << "," 
                        << y0  << "," << x1 << "," << y1 << "]\n";

                    std::string text = class_names[cls_id];
                    text += ":";
                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(2) << score;
                    text += ss.str();

                    int baseline=0;
                    cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

                    cv::rectangle(imgs[batch_id], cv::Point(x0, y0), cv::Point(x1, y1), colors[cls_id], 3);
                    cv::putText(imgs[batch_id], text, cv::Point(x0, y0-5), font_face, font_scale, cv::Scalar(0, 0, 255), thickness, cv::LINE_AA);
                }
                gLogInfo << std::endl;

                std::string out_name = saved_dir + "/det_";

                std::string img_name = img_names[start_id + batch_id];
                size_t index = img_name.find_last_of("/\\");
                if (index != img_name.npos)  {
                    out_name += img_name.substr(index+1);
                } else {
                    out_name += img_name;
                }
                gLogInfo << "Saving the image to " << out_name << std::endl;
                cv::imwrite(out_name, imgs[batch_id]); 
            }
        }
    }

};
} // end of namespace 

// end of this file 
