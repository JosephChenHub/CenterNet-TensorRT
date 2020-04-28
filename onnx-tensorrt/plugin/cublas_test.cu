#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <random>
#include <cassert>
#include <iostream>

using namespace std;

const char* cublasGetErrorString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

cublasHandle_t blas_handle() {
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    const int n = 0;
    //cudaError_t status = cudaGetDevice(&n);
    if(!init[n]) {
        cublasStatus_t st = cublasCreate(&handle[n]);
        if (st != CUBLAS_STATUS_SUCCESS) {
            printf("blas_handle create failed! %s:%d, code:%s\n", __FILE__, __LINE__, cublasGetErrorString(st));
        }
        init[n] = 1;
    }
    return handle[n];
}

template <typename T>
void createBatchBuffers(T* buff[], T* data, const size_t len_per_batch, const int batch_num) {
    for(int i = 0; i < batch_num; ++i) {
        buff[i] = data + len_per_batch * i;
    }
}

void cublas_mat(float* d_C, float* d_A, float *d_B, const int A_ROW, const int A_COL, const int B_COL) {
    cublasHandle_t handle = blas_handle();
    float alpha = 1.0, beta = 0.0;
    cublasStatus_t st = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, B_COL, A_ROW, A_COL,&alpha,
            d_B, B_COL, d_A, A_COL, &beta, d_C, B_COL);

    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemm error occurred! %s : %d, error_code:%s\n", __FILE__, __LINE__, 
                cublasGetErrorString(st));
        exit(-1);
    }

}

#define MAX_BATCH_SIZE 5

void cublas_bmm(float* d_C, float* d_A, float* d_B, 
        const int A_ROW, const int A_COL, const int B_COL, const int batch_num) {
    cublasHandle_t handle = blas_handle();
    float alpha = 1.0, beta = 0.0;

    float* A_buff[MAX_BATCH_SIZE];
    float* B_buff[MAX_BATCH_SIZE];
    float* C_buff[MAX_BATCH_SIZE];
    float** dA_buff, **dB_buff, **dC_buff;
    createBatchBuffers<float>(A_buff, d_A, A_ROW * A_COL, batch_num);
    createBatchBuffers<float>(B_buff, d_B, A_COL * B_COL, batch_num);
    createBatchBuffers<float>(C_buff, d_C, A_ROW * B_COL, batch_num);

    cudaMalloc(&dA_buff, sizeof(float*) * batch_num);
    cudaMalloc(&dB_buff, sizeof(float*) * batch_num);
    cudaMalloc(&dC_buff, sizeof(float*) * batch_num);

    cudaMemcpy(dA_buff, A_buff, sizeof(float*) * batch_num, cudaMemcpyHostToDevice);
    cudaMemcpy(dB_buff, B_buff, sizeof(float*) * batch_num, cudaMemcpyHostToDevice);
    cudaMemcpy(dC_buff, C_buff, sizeof(float*) * batch_num, cudaMemcpyHostToDevice);

    cublasStatus_t st = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            B_COL, A_ROW, A_COL, &alpha, dB_buff, B_COL, 
            dA_buff, A_COL, &beta, dC_buff, B_COL, batch_num);

    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemm error occurred! %s : %d, error_code:%s\n", __FILE__, __LINE__, 
                cublasGetErrorString(st));
        exit(-1);
    }

    cudaFree(dA_buff);
    cudaFree(dB_buff);
    cudaFree(dC_buff);
}

int main() {
    cudaSetDevice(0);
    float* A, *B, *C;
    float* d_A, *d_B, *d_C;
    float* out;
    int A_ROW = 7;
    int A_COL = 9;
    int B_COL = 10;
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(10., 2.);

    size_t A_size = A_ROW * A_COL;
    size_t B_size = A_COL * B_COL;
    size_t C_size = A_ROW * B_COL;
    A = new float[A_size];
    B = new float[B_size];
    C = new float[C_size];
    out = new float[C_size]; 

    cudaMalloc((void**)&d_A, sizeof(float) * A_size);
    cudaMalloc((void**)&d_B, sizeof(float) * B_size); 
    cudaMalloc((void**)&d_C, sizeof(float) * C_size);

    for(int i = 0; i < A_ROW*A_COL; ++i) {
        A[i] = distribution(generator);
    }
    for(int i = 0; i < A_COL*B_COL; ++i) {
        B[i] = distribution(generator);
    }
    float tmp;
    for(int i = 0; i < A_ROW; ++i) {
        for(int j = 0; j < B_COL; ++j) {
            tmp = 0;
            for(int k = 0; k < A_COL; ++k) {
                tmp += A[i * A_COL + k] *  B[k * B_COL + j];
            }
            C[i * B_COL + j] = tmp;
        }
    }

    cudaMemcpy(d_A, A, sizeof(float) * A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * B_size, cudaMemcpyHostToDevice);

    //cublas_mat(d_C, d_A, d_B, A_ROW, A_COL, B_COL);
    cublas_bmm(d_C, d_A, d_B, A_ROW, A_COL, B_COL, 1);

    cudaMemcpy(out, d_C, sizeof(float) * A_ROW * B_COL, cudaMemcpyDeviceToHost);
    /// varification
    for(size_t i = 0; i < C_size; ++i) {
        float err = fabs(out[i] - C[i]);
        if (err > 1e-3) {
            cout << "i:" << i << " out:" << out[i] << " c:" << C[i] << " err:" << err << endl;
        }
        assert(err < 1e-3);
    }
    cout << "cublas test successfully!" << endl;

    delete [] A;
    delete [] B;
    delete [] C;
    delete [] out;
}
