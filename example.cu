#include "common/book.h"
#include "CudaVector.hpp"
#include "VectorSaver.hpp"

#include <iostream>
#include <vector>

#define N 10
__global__ void add( double * a,
                     double * b,
                     double * c )
{
    int tid = blockIdx.x;
    if (tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}

int main(void)
{
    VectorSaver<double> vectorSaver;
    CudaVector<double>  d_a(N);
    CudaVector<double>  d_b(N);
    CudaVector<double>  d_c(N);
    std::vector<double> h_a(N);
    std::vector<double> h_b(N);
    std::vector<double> h_c(N);

    for (int i = 0; i < N; i++)
    {

        h_a[i] = i;
        h_b[i] = i*i;
    }

    d_a.copyToDevice(h_a.data());
    d_b.copyToDevice(h_b.data());
    add<<<N,1>>>( d_a.data(), d_b.data(), d_c.data() );

    d_c.copyToHost(h_c.data());
    for (int i = 0; i < N; i++)
    {
        printf("h_c[%d] = %1.6g\n", i, h_c[i]);
    }

    if (vectorSaver.SaveToFile("vector_data.bin", h_c)) {
        std::cout << "Vector data saved to file successfully." << std::endl;
    } else {
        std::cerr << "Failed to save vector data to file." << std::endl;
    }

    cudaFree(&d_c);
    return 0;
}
