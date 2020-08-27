#pragma once


#include <vector>
#include <cassert>
#include <algorithm>
#include <stddef.h>
#include <cstring>
#include <memory>
#include <math.h>
#include <stdlib.h> // rand
#include <time.h> 
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

#include "sharedmem.hpp"
#include "gpu_common.cuh"
#include "custom.hpp"
#include "decode.hpp"
#include "process.hpp"


template <typename T1, typename T2>
void cuda_warp_affine(const int batch_num, 
        T1* src, const int channel, const int in_h, const int in_w,  
        T2* dst, const int out_h, const int out_w, 
        const float* trans, cudaStream_t stream);

template <typename T>
void fastTopK(T* data, T* buff, const size_t num, const size_t k, T* out);


template <typename T>
class MinHeap {
private:
    int _size;
    T* _val;
    size_t* _idx;
public:
    MinHeap() {
        _size = 0;
        _val = NULL;
        _idx = NULL;
    }
    MinHeap(T* data, const int size) {
        _size = size;
        _idx = new size_t[_size];
        _val = new T[_size];
        memcpy(_val, data, sizeof(T) * _size);
        for(int i = 0; i < _size; ++i) _idx[i] = i;
        this->build_heap();
    }
    ~MinHeap() {
        if (!_size) return;
        delete [] _val;
        delete [] _idx;
    }
    void heapify(const int idx) {
        int left = idx*2 + 1;
        int right = idx*2 + 2;
        int smallest = idx;
        if (left < _size && _val[left] < _val[smallest]) smallest = left;
        if (right < _size && _val[right] < _val[smallest]) smallest = right;
        if (smallest != idx) {
            T tmp = _val[idx];
            _val[idx] = _val[smallest];
            _val[smallest] = tmp;
            size_t tmp2 = _idx[idx];
            _idx[idx] = _idx[smallest];
            _idx[smallest] = tmp2;
            heapify(smallest);
        }
    }
    void build_heap() {
        for(int i = _size / 2 - 1; i >= 0; --i) {
            heapify(i);
        }
    }
    const int size() const {return _size;}
    T* data() {return _val;} 
    size_t* index() {return _idx;}
};

template <typename T>
void top_k(T* in, const size_t num, const size_t k, MinHeap<T>& heap);

// end of this file
