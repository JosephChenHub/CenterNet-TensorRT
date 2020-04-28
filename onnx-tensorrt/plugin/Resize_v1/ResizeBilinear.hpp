/*
 * custom plugin, TensorRT-7
 *
 */ 
#pragma once
#include "NvInfer.h"


#include <cassert>
#include <iostream>


namespace {
    constexpr const char* RESIZE_BILINEAR_PLUGIN_VERSION{"v1"};
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
inline bool is_CHW(nvinfer1::Dims const& dims) {
  return (dims.nbDims == 3 &&
          dims.type[0] == nvinfer1::DimensionType::kCHANNEL &&
          dims.type[1] == nvinfer1::DimensionType::kSPATIAL &&
          dims.type[2] == nvinfer1::DimensionType::kSPATIAL);
}

}

/// inherited from IPluginV2DynamicExt that supports for dynamic inputs'shape 
class MyUpsamplePlugin final: public nvinfer1::IPluginV2DynamicExt {
    std::string _nameSpace;
    nvinfer1::Dims _inputDims;
    nvinfer1::Dims _outputDims;
    nvinfer1::DataType _dataType;

    bool _align_corners = false;
    int _ndims = 2;
    float _scale[nvinfer1::Dims::MAX_DIMS];
public:
    MyUpsamplePlugin(std::vector<float> const& scale, bool const& align_corners)
        :_ndims(scale.size()), _align_corners(align_corners){
        assert(scale.size() <= nvinfer1::Dims::MAX_DIMS);
        std::copy(scale.begin(), scale.end(), _scale);
    }

    MyUpsamplePlugin(const void* data, size_t length) {
        //TODO
        const char *d = reinterpret_cast<const char*>(data), *a = d;
        _align_corners = read<bool>(d);
        _scale[0] = read<float>(d);
        _scale[1] = read<float>(d); 
        _dataType = nvinfer1::DataType::kFLOAT;

        //std::cout << "CONSTRUCTOR:" << _inputDims.d[0]
        //        <<"," << _inputDims.d[1]
        //        <<"," << _inputDims.d[2] <<
        //        " nb:" << _inputDims.nbDims << std::endl;
        //std::cout << " scales:" << _scale[0] << "," << _scale[1]
        //    << std::endl;
        assert(d == a + length);
    }
    ~MyUpsamplePlugin() override=default;
    //MyUpsamplePlugin()=delete; //constructor must has arguments.
  
    nvinfer1::Dims const&  getInputDims(int index) const { return _inputDims; }

    /// override these methods
    int getNbOutputs() const override {return 1; }
    void terminate() override {}
    void destroy() override {delete this; }
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override {return 0;}

    int initialize() override {

      return 0;
    }
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, 
            const nvinfer1::DimsExprs* inputs, int nbInputs, 
                nvinfer1::IExprBuilder& exprBuilder) override { // differ from IPluginV2Ext 

      assert(nbInputs == 1);
      assert(inputs[0].nbDims == 4); 
      assert(outputIndex == 0);

      auto& input = inputs[0];
      nvinfer1::DimsExprs output;
      output.nbDims = 4;

      int N = input.d[0]->getConstantValue();
      int C = input.d[1]->getConstantValue();
      int H = input.d[2]->getConstantValue();
      int W = input.d[3]->getConstantValue();

      int out_h = static_cast<int>(H * _scale[0]);
      int out_w = static_cast<int>(W * _scale[1]);

      output.d[0] = input.d[0];
      output.d[1] = input.d[1];
      output.d[2] = exprBuilder.constant(out_h);
      output.d[3] = exprBuilder.constant(out_w);

       _inputDims.d[0] = N; _inputDims.d[1] = C; _inputDims.d[2] = H; _inputDims.d[3] = W;
      _outputDims.d[0] = N; _outputDims.d[1] = C; _outputDims.d[2] = out_h; _outputDims.d[3] = out_w; 

      return output; 
    }

    int enqueue(const nvinfer1::PluginTensorDesc * inputDesc,
                const nvinfer1::PluginTensorDesc * outputDesc,
                const void* const *  inputs,
                void* const* outputs,
                void* workspace,
                cudaStream_t stream) override;

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, \
            int nbInputs) const override {assert(index == 0); return this->_dataType;}

    size_t getSerializationSize() const override {
        return sizeof(bool) + sizeof(float)*2;
    }
    void serialize(void* buffer) const override {
        char *d = reinterpret_cast<char*>(buffer), *a = d;
        write(d, _align_corners);
        write(d, _scale[0]);
        write(d, _scale[1]); 
        assert(d == a + getSerializationSize());
    }

    nvinfer1::IPluginV2DynamicExt* clone() const override {
        return new MyUpsamplePlugin(*this);
    }
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, 
                        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs
                ) override {  } //do nothing

    /// support format: fp32/fp16 and NCHW
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, \
                int nbInputs, int nbOutputs) override {   
        assert(0 == pos || 1 == pos); // since we only have 1 input and 1 output
        const nvinfer1::PluginTensorDesc& in = inOut[pos];
        if (pos == 0) {
            return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == nvinfer1::TensorFormat::kLINEAR);             
        }
        const nvinfer1::PluginTensorDesc& prev = inOut[pos - 1];
        return in.type == prev.type && in.format == prev.format;
    }

    const char* getPluginType() const override {return RESIZE_BILINEAR_PLUGIN_NAME;}
    const char* getPluginVersion() const override {return RESIZE_BILINEAR_PLUGIN_VERSION;}
    void setPluginNamespace(const char* libNamespace) override {_nameSpace = libNamespace;}
    //const char* getPluginNamespace() const {return _nameSpace.c_str();}
    const char* getPluginNamespace() const {return "";}
    //bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override {return false;}
    //bool canBroadcastInputAcrossBatch(int inputIndex) const override {return false;}
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
    std::vector<float> _scales;
    static nvinfer1::PluginFieldCollection _mFC;
    static std::vector<nvinfer1::PluginField> _mPluginAttributes;
public:
    MyUpsamplePluginCreator() { 
        _mPluginAttributes.emplace_back(nvinfer1::PluginField("align_corners", nullptr, nvinfer1::PluginFieldType::kINT8, 1));
        _mPluginAttributes.emplace_back(nvinfer1::PluginField("scales", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 2));

        _mFC.nbFields = _mPluginAttributes.size();
        _mFC.fields   = _mPluginAttributes.data();
        _scales = std::vector<float>(2, 0.);
        _align_corners = false;
    }
    ~MyUpsamplePluginCreator() {}

    const char* getPluginName() const { return RESIZE_BILINEAR_PLUGIN_NAME; }

    const char* getPluginVersion() const { return RESIZE_BILINEAR_PLUGIN_VERSION; }

    const nvinfer1::PluginFieldCollection* getFieldNames() {
        return &_mFC;
    }

    nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) {
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
                _scales[0] = (static_cast<const float*>(fields[i].data))[0];
                _scales[1] = (static_cast<const float*>(fields[i].data))[1];
            }
        }
        return new MyUpsamplePlugin(_scales, _align_corners);
    }


    nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) { return new MyUpsamplePlugin{serialData, serialLength}; }

    void setPluginNamespace(const char* libNamespace) { mNamespace = libNamespace; }

    const char* getPluginNamespace() const { return mNamespace.c_str(); }

};

/// register plugin
REGISTER_TENSORRT_PLUGIN(MyUpsamplePluginCreator);
// end of this file
