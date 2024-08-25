from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

library_dirs = [torch.utils.cpp_extension.include_paths()[0]]

setup(
    name='shared_buffer_extension',
    ext_modules=[
        CUDAExtension(
            name='shared_buffer_extension',
            sources=['shared_buffer.cpp'],
            library_dirs=library_dirs,
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)