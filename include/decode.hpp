#pragma once





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
        const int batch_num, const int num_classes,  
        const int height, const int width, 
        const int K, const float threshold, 
        const bool reg_exist, const bool cat_spec_wh,
        cudaStream_t stream);

void multi_pose_decode(
        float* det,  
        float* heat, float* wh, float* reg, 
        float* hps, float* hm_hp, float* hp_offset, 
        size_t* heat_ind, size_t* hm_ind, 
        float* inv_trans, 
        const int batch_num, const int num_classes, 
        const int height, const int width, 
        const int K, const float threshold, 
        const bool reg_exist, const bool hm_hp_exist, 
        cudaStream_t stream);
