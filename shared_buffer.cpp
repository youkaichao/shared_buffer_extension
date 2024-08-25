#include "shared_buffer.h"

SharedBufferHolder::SharedBufferHolder(int size_in_bytes) : size_in_bytes(size_in_bytes) {
    size_t N_bytes = size_in_bytes * sizeof(uint8_t);
    CUDA_CALL(cudaHostAlloc((void**)&host_data, N_bytes, cudaHostAllocMapped));
    CUDA_CALL(cudaHostGetDevicePointer((void**)&host_data_in_device, host_data, 0));
}

SharedBufferHolder::~SharedBufferHolder() {
    CUDA_CALL(cudaFreeHost(host_data));
}

torch::Tensor SharedBufferHolder::get_cpu_tensor_view() {
    return torch::from_blob(host_data, {size_in_bytes}, torch::dtype(torch::kUInt8));
}

torch::Tensor SharedBufferHolder::get_cuda_tensor_view() {
    return torch::from_blob(host_data_in_device, {size_in_bytes}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<SharedBufferHolder>(m, "SharedBufferHolder")
        .def(py::init<int>())
        .def("get_cpu_tensor_view", &SharedBufferHolder::get_cpu_tensor_view)
        .def("get_cuda_tensor_view", &SharedBufferHolder::get_cuda_tensor_view);
}