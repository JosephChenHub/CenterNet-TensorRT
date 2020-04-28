#pragma once


template <typename T>
void mergeSort(T* in, const size_t num, T* buff, const bool up=true);

template <typename T>
void bitonicSort(T* in, const size_t num, T* buff, const bool ascending=true);


template <typename T>
void bitonicBatchTopK(T* data, size_t* indices, 
        const int batch_num, const size_t slice_len, const int K, cudaStream_t stream);

/// heat: NxCxHxW, indices: NxCxHxW (topk), classes: NxK
/// output@: det: NxKx6, *num_bbox: int
void ctdet_decode(
        float* det, 
        float* wh, float* reg, 
        float* heat, size_t* indices, 
        float* inv_trans, 
        const int batch_num, const int channel, 
        const int height, const int width, 
        const int K,  const int num_classes, 
        const float threshold, 
        const bool reg_exist, const bool cat_spec_wh,
        cudaStream_t stream);
