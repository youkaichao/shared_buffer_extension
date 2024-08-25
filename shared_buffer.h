#ifndef SHARED_BUFFER_H
#define SHARED_BUFFER_H

#include <torch/extension.h>
#include <cuda_runtime.h>

#define CUDA_CALL(call)                                                          \
  {                                                                              \
    cudaError_t err = (call);                                                    \
    if (err != cudaSuccess) {                                                    \
      std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
      throw std::runtime_error("CUDA error");                                    \
    }                                                                            \
  }

class SharedBufferHolder {
  // This class is used to hold a shared buffer between CPU and GPU.
  // The buffer is allocated on the CPU, via cudaHostAlloc, and then
  // mapped to the GPU.
  // The GPU view can be directly used in cuda kernels.
  // When the kernel first touches the data, the data is transferred
  // to the GPU on demand.

  // This technique is called zero-copy memory mapping with
  // Unified Virtual Address Space (UVA) and is useful when
  // we need to copy many small buffers from CPU to GPU before
  // we launch a kernel.
  // The traditional solution is to launch cudaMemCpyAsync for each
  // buffer, but this is slow when the size of the buffer is small,
  // where the time is dominated by the overhead of launching the
  // cudaMemCpyAsync.

  // With this technique, we can allocate a large buffer on the CPU,
  // and then map the buffer to the GPU. We can then launch a kernel
  // that reads from the GPU view of the buffer. The kernel will
  // implicitly copy the data from the CPU to the GPU on demand,
  // and then process the data as it wants, e.g. copying the data
  // elsewhere in the GPU memory or performing some computation.
public:
    SharedBufferHolder(int size_in_bytes);
    ~SharedBufferHolder();
    torch::Tensor get_cpu_tensor_view();
    torch::Tensor get_cuda_tensor_view();

private:
    uint8_t* host_data;
    uint8_t* host_data_in_device;
    int size_in_bytes;
};

#endif // SHARED_BUFFER_H