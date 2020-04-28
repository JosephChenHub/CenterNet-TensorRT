#include "topk_cpu.hpp"





template <typename T>
void top_k(T* in, const size_t num, const size_t k, MinHeap<T>& heap) {
    /// build a min-heap
    if(heap.size() == 0) heap = MinHeap<T>(in, k);
    /// compare 
    T* heap_ptr = heap.data();
    size_t* heap_idx = heap.index();
    for (size_t i = k; i < num; ++i) {
        if (in[i] > heap_ptr[0])  {
            heap_ptr[0] = in[i];
            heap_idx[0] = i;
            heap.heapify(0);
        }
    }
}


template void top_k<int>(int*, const size_t, const size_t, MinHeap<int>&);
template void top_k<float>(float*, const size_t, const size_t, MinHeap<float>&);
template void top_k<double>(double*, const size_t, const size_t, MinHeap<double>&);
