import glob
import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))
extensions_dir = os.path.join(this_dir, "src")

main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
source_cpp = glob.glob(os.path.join(extensions_dir, "*", "*.cpp"))
source_cuda = glob.glob(os.path.join(extensions_dir, "*", "*.cu"))

if torch.cuda.is_available():
    sources = source_cpp + source_cuda + main_file
else:
    raise Exception('This implementation is only avaliable for CUDA devices.')

setup(
    name='tree_filter',
    version="0.1",
    description="learnable tree filter for pytorch",
    ext_modules=[
        CUDAExtension(
            name='tree_filter_cuda',
            include_dirs=[extensions_dir],
            sources=sources,
            extra_compile_args={'cxx':['-O3'], 'nvcc':['-O3']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

