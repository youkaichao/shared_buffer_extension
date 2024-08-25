import torch
from shared_buffer_extension import SharedBufferHolder

buffer_size = 1024
buffer_holder = SharedBufferHolder(buffer_size)

# this buffer physically resides in the CPU memory
# but we can get a CUDA tensor view of it to be used directly in CUDA kernels
cpu_tensor = buffer_holder.get_cpu_tensor_view().view(dtype=torch.int32)
cuda_tensor = buffer_holder.get_cuda_tensor_view().view(dtype=torch.int32)

d = torch.ones(100, dtype=torch.int32).cuda()

# note here we are modifying the CPU tensor, but the changes will be reflected in the CUDA tensor
cpu_tensor.zero_()
cpu_tensor += 1

out = d + cuda_tensor[:100]

# this will print 2
print(out)