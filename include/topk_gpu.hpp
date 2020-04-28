#ifndef __TOPK_GPU_HPP
#define __TOPK_GPU_HPP

#include <cuda_runtime_api.h>






template <typename T>
void fastTopK(T* data, T* buff, const size_t num, const size_t k, T* out);


#endif 
