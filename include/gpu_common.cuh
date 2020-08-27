#ifndef __GPU_COMMON_CUH
#define __GPU_COMMON_CUH

#include <utility>
#include <unordered_map>
#include <iostream>
#include <cuda_runtime_api.h>

template <typename T>
struct DoubleBuffer {
    int selector;
    T* buffers[2];
    __host__ __device__ __forceinline__ DoubleBuffer() {
        selector = 0;
        buffers[0] = NULL;
        buffers[1]=NULL;
    }

     __host__ __device__ __forceinline__ DoubleBuffer(T* d_current, T* d_alt) {
         selector = 0;
         buffers[0] = d_current;
         buffers[1] = d_alt;
     }
     __host__ __device__ __forceinline__ T* Current() {
         return buffers[selector];
     }
     __host__ __device__ __forceinline__ T* Alternate () {
         return buffers[selector^1];
     }
};
/// A simple pair type for CUDA device usage
template <typename K, typename V>
struct Pair {
    __host__ __device__ __forceinline__ Pair() {}

    __host__ __device__ __forceinline__ Pair(K key, V value)
        : k(key), v(value) {  }

    __host__ __device__ __forceinline__ bool
        operator==(const Pair<K, V>& rhs) const {
            return (k == rhs.k) && (v == rhs.v);
        }

    __host__ __device__ __forceinline__ bool
        operator!=(const Pair<K, V>& rhs) const {
            return !operator==(rhs);
        }

    __host__ __device__ __forceinline__ bool
        operator<(const Pair<K, V>& rhs) const {
            return (k < rhs.k) || ((k == rhs.k) && (v < rhs.v));
        }

    __host__ __device__ __forceinline__ bool
        operator>(const Pair<K, V>& rhs) const {
            return (k > rhs.k) || ((k == rhs.k) && (v > rhs.v));
        }

    __host__ __device__ Pair& operator=(const Pair<K, V>& rhs) {
        this->k = rhs.k;
        this->v = rhs.v;
        return *this;
    }

    K k;
    V v;
};


#define CHECK_CUDA(e) { if(e != cudaSuccess) { \
    printf("cuda failure: %s:%d: '%s'\n", __FILE__, __LINE__, \
            cudaGetErrorString(e)); \
        exit(0); \
    } \
}

#define CHECK_LAST_ERR(func) { \
    cudaError_t e = cudaGetLastError();\
    if (e != cudaSuccess) {\
        printf("cuda failure of %s: %s:%d: '%s'\n", func, __FILE__, __LINE__, \
            cudaGetErrorString(e)); \
        exit(-1); \
    } \
}

///
/// \brief a Simple GPU memory management 
/// 
class GPUAllocator {
private:
   std::unordered_map<std::string, std::pair<void*, size_t>> _data;
  
public:
   GPUAllocator() {}
   ~GPUAllocator() {
       for(auto &e: _data) {
           auto ptr = e.second.first;
           if(ptr) CHECK_CUDA(cudaFree(ptr));
       }
   }

   template <typename T>
   T* get_pointer(const char* name) const { /// note that the pointer is a memory address on GPU, you cannot directly access the contents by reference it 
       auto res = _data.find(std::string(name));
       if(res == _data.end()) return nullptr;
       return reinterpret_cast<T*>(res->second.first);
   }

   template <typename T>
   T* allocate(const char* name, size_t size) {
       T* tmp;
       auto res = _data.find(std::string(name));
       if (res == _data.end() || res->second.second < size) {
	    if (res != _data.end()) {
                std::cout << "Warning: existing memory :" << name 
                    << " is smaller than the requested, and it will be destroyed and allocated again!" 
                    << std::endl;
		auto ptr = res->second.first;
		if(ptr) CHECK_CUDA(cudaFree(ptr));
            }
            CHECK_CUDA(cudaMalloc((void**)&tmp, sizeof(T) * size));
	    auto item = std::make_pair(reinterpret_cast<void*>(tmp), size);
	    if (res == _data.end()) _data.emplace(std::string(name),  item);
	    else res->second = item; 
       } else {
            tmp = reinterpret_cast<T*>(res->second.first); 
       }
       return tmp;
   }


};



#endif
