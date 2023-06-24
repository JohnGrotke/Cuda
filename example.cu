#include "common/book.h"
#include "CudaVector.hpp"
#include "VectorSaver.hpp"
#include <sys/time.h>

#include <iostream>
#include <vector>

#define N 10000000

#define N_THREADS 256

#define N_RUNS 200

__global__ void add_gpu( double * a,
                         double * b,
                         double * c )
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

double add_cpu( std::vector<double> & a,
               std::vector<double> & b,
               std::vector<double> & c )
{
    for ( int i = 0; i < N; ++i )
    {
        c[i] = a[i] + b[i];
    }
    
    return true;
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
    std::vector<double> cpu_times;
    std::vector<double> cuda_times_copy;
    std::vector<double> cuda_times_no_copy;

    struct timeval t1_cpu, t2_cpu, t1_cuda, t2_cuda;
    double time_cpu;
    double time_cuda;
    int n_blocks = N + (N_THREADS - 1) / N_THREADS;

    for (int i = 0; i < N; i++)
    {
        h_a[i] = i;
        h_b[i] = i + 1;
    }

    // CPU Test
    for ( int i_run = 0; i_run < N_RUNS; i_run++ )
    {
        gettimeofday(&t1_cpu, 0);
        add_cpu( h_a, h_b, h_c );
        gettimeofday(&t2_cpu, 0);

        time_cpu = (1000000.0*(t2_cpu.tv_sec-t1_cpu.tv_sec) + t2_cpu.tv_usec-t1_cpu.tv_usec)/1000.0;
        cpu_times.push_back(time_cpu);
    }

    // GPU Test With Copying
    for ( int i_run = 0; i_run < N_RUNS; i_run++ )
    {
        gettimeofday(&t1_cuda, 0);

        d_a.copyToDevice(h_a.data());
        d_b.copyToDevice(h_b.data());

        add_gpu<<<n_blocks,N_THREADS>>>( d_a.data(), d_b.data(), d_c.data() );

        d_c.copyToHost(h_c.data());

        gettimeofday(&t2_cuda, 0);
        
        time_cuda = (1000000.0*(t2_cuda.tv_sec-t1_cuda.tv_sec) + t2_cuda.tv_usec-t1_cuda.tv_usec)/1000.0;
        cuda_times_copy.push_back(time_cuda);
    }

    // GPU Test No Copying
     for ( int i_run = 0; i_run < N_RUNS; i_run++ )
     {
         gettimeofday(&t1_cuda, 0);
         add_gpu<<<n_blocks,N_THREADS>>>( d_a.data(), d_b.data(), d_c.data() );
 
         d_c.copyToHost(h_c.data());
 
         gettimeofday(&t2_cuda, 0);
         
         time_cuda = (1000000.0*(t2_cuda.tv_sec-t1_cuda.tv_sec) + t2_cuda.tv_usec-t1_cuda.tv_usec)/1000.0;
         cuda_times_no_copy.push_back(time_cuda);
     }

    printf("Time to run CPU:\n");
    double cpu_total_time = 0.0;
    double cuda_with_copying_total_time = 0.0;
    double cuda_without_copying_total_time = 0.0;
    for ( int i_run = 0; i_run < N_RUNS; i_run++ )
    {
        printf("Run %d:  %1.6g ms \n", i_run, cpu_times[i_run]);
        cpu_total_time += cpu_times[i_run];
    }
    printf("Total CPU time: %1.6g\n\n", cpu_total_time);

    printf("Time to run CUDA with copying:\n");
    for ( int i_run = 0; i_run < N_RUNS; i_run++ )
    {
        printf("Run %d:  %1.6g ms \n", i_run, cuda_times_copy[i_run]);
        cuda_with_copying_total_time += cuda_times_copy[i_run];
    }
    printf("Total CUDA with copying time: %1.6g\n\n", cuda_with_copying_total_time);

    printf("Time to run CUDA without copying:\n");
    for ( int i_run = 0; i_run < N_RUNS; i_run++ )
    {
        printf("Run %d:  %1.6g ms \n", i_run, cuda_times_no_copy[i_run]);
        cuda_without_copying_total_time += cuda_times_no_copy[i_run];
    }
    printf("Total CUDA without copying time: %1.6g\n\n", cuda_without_copying_total_time);

    // Serialize data to file
    if (vectorSaver.SaveToFile("vector_data.bin", h_c)) {
        std::cout << "Vector data saved to file successfully." << std::endl;
    } else {
        std::cerr << "Failed to save vector data to file." << std::endl;
    }

    cudaFree(&d_c);
    return 0;
}
