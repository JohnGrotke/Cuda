#include <iostream>
#include <cuda_runtime.h>

template <typename T>
class CudaVector {
public:
    CudaVector() : size_(0), data_(nullptr), stream_(0), ownsStream_(true) {}

    explicit CudaVector(size_t size, cudaStream_t stream = 0) : size_(size), stream_(stream), ownsStream_(false) {
        cudaMalloc((void**)&data_, size_ * sizeof(T));
    }

    ~CudaVector() {
        cudaFree(data_);
        if (ownsStream_) {
            cudaStreamDestroy(stream_);
        }
    }

    void resize(size_t size) {
        if (data_) {
            cudaFree(data_);
        }
        size_ = size;
        cudaMalloc((void**)&data_, size_ * sizeof(T));
    }

    size_t size() const {
        return size_;
    }

    void copyToDevice(const T* hostData) {
        cudaMemcpy(data_, hostData, size_ * sizeof(T), cudaMemcpyHostToDevice);
        cudaStreamSynchronize(stream_);
    }

    void copyToHost(T* hostData) const {
        cudaMemcpy(hostData, data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(stream_);
    }

    void copyToDeviceAsync(const T* hostData) {
        cudaMemcpyAsync(data_, hostData, size_ * sizeof(T), cudaMemcpyHostToDevice, stream_);
    }

    void copyToHostAsync(T* hostData) const {
        cudaMemcpyAsync(hostData, data_, size_ * sizeof(T), cudaMemcpyDeviceToHost, stream_);
    }

    void wait() const {
        cudaStreamSynchronize(stream_);
    }

    T* data() {
        return data_;
    }

    const T* data() const {
        return data_;
    }

private:
    size_t size_;
    T* data_;
    cudaStream_t stream_;
    bool ownsStream_;
};