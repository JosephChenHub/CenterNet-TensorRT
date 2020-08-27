#pragma once 

#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>




void log2_series(unsigned int value, std::vector<int>& param);
int log2_32 (unsigned int value);

void get_affine_transform(
        float* mat_data, 
        const float* const center, \
        const float* const scale, \
        const float* const shift, 
        const float rot, \
        const int output_h, const int output_w, 
        const bool inv=false);

std::vector<std::string> split_str(std::string& str, std::string pattern);
void makedirs(const char* dir);
