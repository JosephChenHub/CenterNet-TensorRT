#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <fstream>
#include <memory>
#include "logger.h"
#include "NvInferVersion.h"
#include "gpu_common.cuh"

namespace TRT {
    using namespace std;
    using namespace nvinfer1;
class TRTInferBase {
private:
    struct NvInferDeleter {
        template <typename T>
        void operator()(T* obj) const {
            if (obj) obj->destroy();
        }
    };
    // note that the context must be destroyed before the engine (follow this order). 
    unique_ptr<IRuntime, NvInferDeleter> _runtime;
    unique_ptr<ICudaEngine, NvInferDeleter> _engine;
    unique_ptr<IExecutionContext, NvInferDeleter> _context;

    unordered_map<string, int> _name_index;

protected:
    unique_ptr<GPUAllocator> _gpu_mem;
    bool _dynamic; 
    bool _mem_initialized; 
    void* _d_imgs; // cpu mem. -> gpu mem.
    float* _buffers[32]; 
    string _input_name;
    vector<string> _binding_names; 
    map<string, pair<int, size_t>> _engine_name_size;
    int _nb_bindings;  // number of inputs and outputs nodes
    int _runtime_input_dims[4]; // inputs' dimension 
public:
    TRTInferBase () = delete; 
    TRTInferBase(const char* engine_name, const int device_id=0, 
            const bool dynamic=false, const char* input_name = "input"):
        _nb_bindings(0), _dynamic(dynamic), _mem_initialized(false) {

	_gpu_mem = unique_ptr<GPUAllocator>(new GPUAllocator); 
        string trt_file(engine_name);
        vector<char> trtModelStream_;
        size_t size(0);
        gLogInfo << "Loading engine file:" << trt_file << endl;
        ifstream file(trt_file, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream_.resize(size);
            file.read(trtModelStream_.data(), size);
            file.close();
        } else {
            gLogFatal << "Failed to open the engine file:" << trt_file << endl;
            exit(-1);
        }        
        gLogInfo << " trt file's size: " << size << endl;
        // set device id before obtaining the engine 
        CHECK_CUDA(cudaSetDevice(device_id));
        // deserialize the engine 
        _runtime = unique_ptr<IRuntime, NvInferDeleter>(createInferRuntime(gLogger));
        _engine = unique_ptr<ICudaEngine, NvInferDeleter>(_runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr));
        if(!_engine) {
            gLogFatal << " Failed to create the engine from .trt file!" << endl;
            exit(-1);
        } else {
            gLogInfo << " Create the engine from " << trt_file << " successfully!" << endl;
        }
        _context = unique_ptr<IExecutionContext, NvInferDeleter>(_engine->createExecutionContext());
        if (!_context) {
            gLogFatal << "Failed to create the execution context!" << endl;
            exit(-1);
        }
        // inquiry the dimensions 
        _nb_bindings = _context->getEngine().getNbBindings();
        _binding_names.resize(_nb_bindings);
        for(int i = 0; i < _nb_bindings; ++i) {
            auto dim = _context->getEngine().getBindingDimensions(i);
            string name = _context->getEngine().getBindingName(i);
            int index = _context->getEngine().getBindingIndex(name.c_str()); 
            assert (index == i);
            _name_index.emplace(name, index);

            bool flag = (dim.d[0] == -1 ) || (dim.d[1] == -1) || (dim.d[2] == -1) || (dim.d[3] == -1);
            if (flag != _dynamic) {
                gLogFatal << "The constructor use dynamic shape :" << flag << " whereas the engine's setting is :" << flag << endl;
                exit(-1);
            }
            if (flag && NV_TENSORRT_MAJOR < 7 ) {
                gLogFatal << "Dynamic shape was introduced after TensorRT 7." << endl;
                exit(-1);
            }
            size_t buff_size = dim.d[0] * dim.d[1] * dim.d[2] * dim.d[3];
            size_t pos = name.find(input_name);
            if (pos != name.npos) {
                _input_name = name;
                _runtime_input_dims[0] = dim.d[0];
                _runtime_input_dims[1] = dim.d[1];
                _runtime_input_dims[2] = dim.d[2];
                _runtime_input_dims[3] = dim.d[3];
            }

            gLogInfo << "tensor's name:" << name 
                << " dim:(" << dim.d[0] << ","
                << dim.d[1] << ","
                << dim.d[2] << ","
                << dim.d[3] << ","
                << " size:" << buff_size << endl;

            _binding_names[i] = name;
            _engine_name_size.emplace(name, make_pair(i, size));
        }
        if (_input_name.empty()) {
            gLogFatal << "Please check the input node's name!" << endl;
            exit(-1);
        }
    }
    virtual ~TRTInferBase() { };


    // use this native pointer to transfer the raw data, then call the infer function
    template <typename T>
    T* data_pointer() {
        return reinterpret_cast<T*>(_d_imgs); 
    }
    
    // inputs' type maybe uint8_t, or float
    void infer(cudaStream_t stream) {
        if(!_d_imgs || !_mem_initialized) {
            gLogFatal << "Some memory has not been initialized!" << endl;
            exit(-1);
        }
        pre_process(stream);
        _context->enqueue(_runtime_input_dims[0], (void**)_buffers, stream, nullptr);
        post_process(stream);
        cudaStreamSynchronize(stream);
    }

    // for user's inquiry
    int getBindingIndex(const char* name) const { 
        auto res = _name_index.find(name);
        if (res == _name_index.end()) {
            gLogFatal << "node's name : " << name << " does not exists!" << endl;
            exit(-1);
        }
        return res->second;
    }
    int nbBindings () const { return _nb_bindings; }
    int batchSize() const {return _runtime_input_dims[0]; }
    int channel () const {return _runtime_input_dims[1];}
    int height () const {return _runtime_input_dims[2];}
    int width () const {return _runtime_input_dims[3];}

    // override these methods
    virtual size_t raw_image_size() const = 0;

    virtual void visualize(int argc, void* argv[]) = 0;
protected:
    virtual void pre_process(cudaStream_t stream) = 0;

    virtual void post_process(cudaStream_t stream) = 0;

    virtual void mem_alloc() = 0; // _mem_initialized must be set true after calling this

}; 
} // end of namespace : TRT

// end of this file
