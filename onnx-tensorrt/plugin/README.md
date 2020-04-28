This is an implementation of some custom plugins using TensorRT API, CUDA API, and c++. Each sub-module is equipped with gtest samples. 
# Dependencies
- CUDA 10.0
- TensorRT-7.0.0.11 (for CUDA10.0)
- CUDNN (for CUDA10.0, may not be used)
- libtorch (torch c++ lib of cpu version, gpu version may conflict with the environment)
- gtest (Google C++ testing framework)

# Current plugins:
- Resize_v0 : upsample_bilinear2d, inherited from IPluginV2Ext (introduced in TensorRT 5.1)
- Resize_v1 : **TODO** upsample_bilinear2d, inherited from IPluginV2DynamicExt
- DCN: deformable CNN

# Build
see bash
```
compile.sh
```
To test **cublas** apis, compile the `cublas_test.cu` with this command
```
nvcc -Xcompiler -fPIC -std=c++11 -lcublas cublas_test.cu -o cublas_test &&
./cublas_test
```

# Precaution:
For debug mode, since the libtorch I used was compiled beforehanded, the lib may confict with your own environment.  Please do these operations:
- download and unzip the libtorch lib: [libtorch-1.2.0](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.2.0.zip), which was compiled beforehanded with new GCC ABI.
- install the Googletest: https://github.com/google/googletest.git . 
- remove the `libgtest.a` and `libgtest_main.a` in libtorch to avoid conflicts.

For release mode,  as u see in the `CMakeLists.txt`, we do not need to  link the libtorch and gtest. 
