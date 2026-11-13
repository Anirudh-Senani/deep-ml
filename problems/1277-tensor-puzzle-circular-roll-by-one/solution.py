#include <cuda_runtime.h>
#include <vector>

__global__ void roll_kernel(const float* a, float* out, int n) {
    // out[i] = a[(i+1) % n]
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    out[i] = a[(i+1) % n];
}

std::vector<float> roll(const std::vector<float>& a) {
    int n = a.size();
    std::vector<float> out(n);
    size_t size = n * sizeof(float);

    float *da, *dout;
    cudaMalloc((void**)&da, size);
    cudaMalloc((void**)&dout, size);

    cudaMemcpy(da, a.data(), size, cudaMemcpyHostToDevice);

    int threads = 256;
    dim3 threadsPerBlock(threads);
    int blocks = (n + threads - 1)/threads;
    dim3 numBlocks(blocks);
    roll_kernel<<<numBlocks, threadsPerBlock>>>(da, dout, n);

    cudaDeviceSynchronize();

    cudaMemcpy(out.data(), dout, size, cudaMemcpyDeviceToHost);

    cudaFree(da);
    cudaFree(dout);

    return out;
}
