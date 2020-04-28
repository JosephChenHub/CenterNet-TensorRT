#include "ResizeBilinear.hpp"

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include <Python.h>
#include <pybind11/pybind11.h>

PYBIND11_MODULE(MyUpsample, m)
{
    namespace py = pybind11;

    py::module::import("tensorrt");

    py::class_<nvinfer1::IPluginV2Ext, std::unique_ptr<nvinfer1::IPluginV2Ext, py::nodelete>>(m, "IPluginV2Ext");  // <-- Addition as per pybind:#172

  
    // Note that we only need to bind the constructors manually. Since all other methods override IPlugin functionality, they will be automatically available in the python bindings.
    // The `std::unique_ptr<MyUpsamplePlugin, py::nodelete>` specifies that Python is not responsible for destroying the object. This is required because the destructor is private.
    py::class_<MyUpsamplePlugin, nvinfer1::IPluginV2Ext, std::unique_ptr<MyUpsamplePlugin, py::nodelete>>(m, "MyUpsample")
    .def(py::init<std::vector<float>const&, bool const&>())
    .def(py::init<const void*, size_t>());

    py::class_<nvinfer1::IPluginCreator>(m, "IPluginCreator");

    py::class_<MyUpsamplePluginCreator, nvinfer1::IPluginCreator>(m, "MyUpsamplePluginCreator", py::multiple_inheritance())
    .def(py::init<>());
}
