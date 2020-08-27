/*
 * custom plugin, TensorRT-7
 *
 */ 
#pragma once
#include "NvInfer.h"

#include <thread>
#include <vector>
#include <cassert>
#include <iostream>


namespace ResizeBuff{
    constexpr const char* RESIZE_BILINEAR_PLUGIN_VERSION{"v0"};
    constexpr const char* RESIZE_BILINEAR_PLUGIN_NAME{"MyUpsample"};

    //Write values into buffer
    template <typename T>
    void write(char*& buffer, const T& val) {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    // Read values from buffer
    template <typename T>
    T read(const char*& buffer) {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

}


/// inherited from IPluginV2Ext 
class MyUpsamplePlugin final: public nvinfer1::IPluginV2Ext {
private:
    std::string _nameSpace;
    nvinfer1::Dims _inputDims;
    nvinfer1::Dims _outputDims;
    nvinfer1::DataType _dataType;
    bool _align_corners;
    int _ndims;
    float _scale[2];
public:
    MyUpsamplePlugin(std::vector<float> const& scale, bool const& align_corners)
        :_ndims(scale.size()), _align_corners(align_corners){
        assert(scale.size() <= nvinfer1::Dims::MAX_DIMS);
        _scale[0] = scale[0];
        _scale[1] = scale[1];
        assert(_scale[0] > 0 && _scale[1] > 0);
    }

    MyUpsamplePlugin(const void* data, size_t length) {
        using namespace ResizeBuff;
        const char *d = reinterpret_cast<const char*>(data), *a = d;
        _align_corners = read<bool>(d);
        _inputDims.nbDims = 3;
        _scale[0] = read<float>(d);
        _scale[1] = read<float>(d); 
        _inputDims = nvinfer1::Dims3();
        _inputDims.d[0] = read<int>(d);
        _inputDims.d[1] = read<int>(d);
        _inputDims.d[2] = read<int>(d);
        _outputDims = nvinfer1::Dims3();
        _outputDims.d[0] = read<int>(d);
        _outputDims.d[1] = read<int>(d);
        _outputDims.d[2] = read<int>(d);

        _dataType = nvinfer1::DataType::kFLOAT;
        assert(_scale[0] > 0 && _scale[1] > 0);
        assert(d == a + length);
        //std::cout << "**** MyUpsamplePlugin Constructor2 has been called!" << std::endl;
    }
    ~MyUpsamplePlugin() override=default;
    MyUpsamplePlugin()=delete; //constructor must has arguments.
    bool align_corners() const {return _align_corners;}
    std::vector<float> scales() const {return {_scale[0], _scale[1]}; }
    nvinfer1::Dims const&  getInputDims(int index) const { return _inputDims; }

    /// override these methods
    int getNbOutputs() const override {return 1; }
    void terminate() override {}
    void destroy() override { delete this; }
    size_t getWorkspaceSize(int) const override {return 0;}

    int initialize() override {
      //std::cout << "**** initialize called! **** id:"  << this<<  std::endl;

      return 0;
    }

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims *inputDims, int nbInputs) override {
      //std::cout << "**** getOutputDimensions called! **** id:" << this << std::endl;
      assert(index == 0);
      assert(nbInputs == 1);
      auto& input = inputDims[0];
      assert(3 == input.nbDims);  /// CHW
      assert(input.d[0] > 0 && input.d[1] > 0 && input.d[2] > 0 );
      assert(_scale[0] > 0 && _scale[1] > 0);

      nvinfer1::Dims output;
      output.nbDims = input.nbDims;
      //std::cout << "#### input.nbDims:" << input.nbDims << " shape:(";
      for(int i = 0; i < input.nbDims; ++i) {
          std::cout << input.d[i] << ",";
      }
      std::cout << ")" << std::endl;

      //std::cout << "###scale:" << _scale[0] << "," << _scale[1] << std::endl;

      for( int d=0; d < input.nbDims; ++d ) {
        output.type[d] = input.type[d];
        if(d == input.nbDims-2) {
          output.d[d] = static_cast<int>(input.d[d] * _scale[0]);
        } else if(d == input.nbDims-1) {
          output.d[d] = static_cast<int>(input.d[d] * _scale[1]);
        }
        else {
          output.d[d] = input.d[d];
        }
      }
      assert(output.d[0] > 0 && output.d[1] > 0 && output.d[2] > 0);
      this->_inputDims.nbDims = 3;
      this->_outputDims.nbDims = 3;
      for(int i = 0; i < 3; ++i) {
        this->_inputDims.d[i] = input.d[i];
        this->_outputDims.d[i] = output.d[i];
      }
      //printf("Inputs & outputs'shape: (%d, %d, %d), (%d, %d, %d)", _inputDims.d[0], _inputDims.d[1], _inputDims.d[2], _outputDims.d[0], _outputDims.d[1], _outputDims.d[2]);
      return output;
    }

    int enqueue(int batch_size, const void* const* inputs, \
            void** outputs, void* workspace, cudaStream_t stream) override;

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, \
            int nbInputs) const override {assert(index == 0); 
        //return this->_dataType;
        return nvinfer1::DataType::kFLOAT;
    }

    size_t getSerializationSize() const override {
        return sizeof(bool) + sizeof(float)*2 + sizeof(int) * 6;
    }
    /// serialize the engine
    void serialize(void* buffer) const override {
        using namespace ResizeBuff;
        char *d = reinterpret_cast<char*>(buffer), *a = d;
        write(d, _align_corners);
        write(d, _scale[0]);
        write(d, _scale[1]); 
        write(d, _inputDims.d[0]);
        write(d, _inputDims.d[1]);
        write(d, _inputDims.d[2]);
        write(d, _outputDims.d[0]);
        write(d, _outputDims.d[1]);
        write(d, _outputDims.d[2]);

        assert(d == a + getSerializationSize());
    }

    nvinfer1::IPluginV2Ext* clone() const override {
        return new MyUpsamplePlugin(*this);
    }
    void configurePlugin(const nvinfer1::Dims* inputDims, int nbInputs, \
            const nvinfer1::Dims* outputDims, int nbOutputs, \
            const nvinfer1::DataType* inputTypes, const nvinfer1::DataType* outputTypes,\
            const bool* inputIsBroadcast, const bool* outputIsBroadcast, \
            nvinfer1::PluginFormat floatFormat, int maxBatchSize) override {
        bool format = supportsFormat(inputTypes[0], floatFormat);
        assert(format);
        this->_inputDims = inputDims[0];
        this->_dataType = outputTypes[0];
        this->_outputDims = outputDims[0];
        std::cout << "***configurePlugin called!*** id:" << this << std::endl;
    }

    /// support format: fp32/fp16 and NCHW
    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override {
        //return ((type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF) 
        return (type == nvinfer1::DataType::kFLOAT  
                && format == nvinfer1::PluginFormat::kNCHW);
    }
    const char* getPluginType() const override {return ResizeBuff::RESIZE_BILINEAR_PLUGIN_NAME;}
    const char* getPluginVersion() const override {return ResizeBuff::RESIZE_BILINEAR_PLUGIN_VERSION;}
    void setPluginNamespace(const char* libNamespace) override {_nameSpace = libNamespace;}
    //const char* getPluginNamespace() const {return _nameSpace.c_str();}
    const char* getPluginNamespace() const {return "";}
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override {return false;}
    bool canBroadcastInputAcrossBatch(int inputIndex) const override {return false;}
    void attachToContext(
            cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator
            ) override {}
    void detachFromContext() override {}

};

/// IPluginCreator
class MyUpsamplePluginCreator : public nvinfer1::IPluginCreator {
private:
    std::string mNamespace;
    bool _align_corners;
    float _scale[2];
    static nvinfer1::PluginFieldCollection _mFC;
    static std::vector<nvinfer1::PluginField> _mPluginAttributes;
public:
    MyUpsamplePluginCreator() { 
        _mPluginAttributes.emplace_back(nvinfer1::PluginField("align_corners", nullptr, nvinfer1::PluginFieldType::kINT8, 1));
        _mPluginAttributes.emplace_back(nvinfer1::PluginField("scales", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 2));

        _mFC.nbFields = _mPluginAttributes.size();
        _mFC.fields   = _mPluginAttributes.data();
        _scale[0] = 0; _scale[1] = 0;
        _align_corners = false;
    }
    ~MyUpsamplePluginCreator() {}

    const char* getPluginName() const { return ResizeBuff::RESIZE_BILINEAR_PLUGIN_NAME; }

    const char* getPluginVersion() const { return ResizeBuff::RESIZE_BILINEAR_PLUGIN_VERSION; }

 nvinfer1::PluginFieldCollection* getFieldNames() {
        return &_mFC;
    }

    nvinfer1::IPluginV2Ext* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) {
        const nvinfer1::PluginField* fields = fc->fields;
        assert (fc->nbFields == 2);
        for (int i = 0; i < fc->nbFields; ++i) {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "align_corners")) {
                assert(fields[i].type == nvinfer1::PluginFieldType::kINT8);
                int align = *(static_cast<const int*>(fields[i].data));
                _align_corners = align ? true:false;
            }
            if(!strcmp(attrName, "scales")) {
                assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
                _scale[0] = (static_cast<const float*>(fields[i].data))[0];
                _scale[1] = (static_cast<const float*>(fields[i].data))[1];
            }
        }
        assert(_scale[0] > 0 && _scale[1] > 0);
        return new MyUpsamplePlugin(_scale, _align_corners);
    }


    nvinfer1::IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) { 
        return new MyUpsamplePlugin(serialData, serialLength); 
    }

    void setPluginNamespace(const char* libNamespace) { mNamespace = libNamespace; }

    const char* getPluginNamespace() const { return mNamespace.c_str(); }

};

/// register plugin
REGISTER_TENSORRT_PLUGIN(MyUpsamplePluginCreator);
// end of this file
