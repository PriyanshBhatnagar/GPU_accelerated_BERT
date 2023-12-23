#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include<iostream>

using namespace std;

__global__ void kernel_multiply(float* A, float* B, float* C, int M, int N, int K);

void cu_multiply(float* A, float* B, float* C, int M, int N, int K, int X)
{
    // Assume A = XxNxK, B = KxM matrix, C = XxNxM
    float* d_a, * d_b, * d_c;
    cudaStream_t stream[20];

    dim3 blk(32, 32, 1);
    dim3 grid((M + blk.x - 1) / blk.x, (N + blk.y - 1) / blk.y, 1);

    int size_a = X * N * K;
    int size_b = K * M;
    int size_c = X * N * M;

    cudaMalloc((void**)&d_a, size_a * sizeof(float));
    cudaMalloc((void**)&d_b, size_b * sizeof(float));
    cudaMalloc((void**)&d_c, size_c * sizeof(float));

    int segmentSize = N * K;

    for (int i = 0; i < X; ++i)
    {
        cudaStreamCreate(&stream[i]);
    }

    for (int i = 0; i < X; ++i)
    {
        int offset = i * segmentSize;
        cudaMemcpyAsync(&d_a[offset], &A[offset], segmentSize * sizeof(float), cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&d_b[offset], &B[offset], M * K * sizeof(float), cudaMemcpyHostToDevice, stream[i]);
        kernel_multiply << <grid, blk, 0, stream[i] >> > (d_a + offset, d_b, d_c + i*M*N, M, N, K);
        cudaMemcpyAsync(&C[offset], &d_c[offset], N * M * sizeof(float), cudaMemcpyDeviceToHost, stream[i]);

    }

    for (int i = 0; i < X; ++i)
    {
        cudaStreamSynchronize(stream[i]);
        cudaStreamDestroy(stream[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}


__global__ void kernel_multiply(float* A, float* B, float* C, int M, int N, int K)
{
    int I = blockIdx.y * blockDim.y + threadIdx.y;
    int J = blockIdx.x * blockDim.x + threadIdx.x;

    if ((I < N) && (J < M))
    {
        {
            float _c = 0;
            for (unsigned int k = 0; k < K; k++)
            {
                float a = A[I * K + k];
                float b = B[k * M + J];
                _c += a * b;
            }
            C[I * M + J] = _c;
        }
    }
}

/*
* 
* 
* The code below is an implementation of matrix multiplication on CPU. 

// CPU function to multiply matrices
void cpu_multiply(float* A, float* B, float* C, int M, int N, int K, int X) {
    for (int x = 0; x < X; x++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                float sum = 0;
                for (int k = 0; k < K; k++) {
                    sum += A[x * N * K + i * K + k] * B[k * M + j];
                }
                C[x * N * M + i * M + j] = sum;
            }
        }
    }
}

// Function to compare results
bool compare_matrices(float* A, float* B, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(A[i] - B[i]) > 1e-2) {
            return false;
        }
    }
    return true;
}


int main() {
    //const int N = 128, M = 768, K = 768, X = 10;
    const int N = 128, M = 768, K = 768, X = 1;
    int size_a = N * K * X;
    int size_b = K * M;
    int size_c = N * M * X;

    float* A, * B, * C_gpu, * C_cpu;

    // Allocate memory
    A = (float*)malloc(size_a * sizeof(float));
    B = (float*)malloc(size_b * sizeof(float));
    C_gpu = (float*)malloc(size_c * sizeof(float));
    C_cpu = (float*)malloc(size_c * sizeof(float));

    // Initialize matrices A and B
    for (int i = 0; i < size_a; i++) A[i] = rand() % 100 / 100.0f;
    for (int i = 0; i < size_b; i++) B[i] = rand() % 100 / 100.0f;

    /*
    for (int i = 0; i < size_a; i++) {
        A[i] = (float)i/1000;
    }
    for (int i = 0; i < size_b; i++)
    {
        B[i] = (float)i/10;
    }
    

    // GPU multiplication
    cu_multiply(A, B, C_gpu, M, N, K, X);

    // CPU multiplication
    cpu_multiply(A, B, C_cpu, M, N, K, X);

    // Compare results
    if (compare_matrices(C_gpu, C_cpu, size_c)) {
        printf("GPU and CPU results match.\n");
    }
    else {
        printf("Mismatch between GPU and CPU results!\n");
    }

    /*
    for (int i = 0; i < size_a; i++)
    {
        std::cout << A[i] << " ";

    }
    cout << "\n";

    for (int i = 0; i < size_c; i++)
    {
        std::cout << C_cpu[i] << " ";

    }

    cout << "\n";

    for (int i = 0; i < size_c; i++)
    {
        std::cout << C_gpu[i] << " ";
    }
    

    // Free memory
    free(A);
    free(B);
    free(C_gpu);
    free(C_cpu);

    return 0;
}

*/