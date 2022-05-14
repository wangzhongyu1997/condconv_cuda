from setuptools import setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

print(cpp_extension.include_paths()) 

setup(
    name='condconv',
    ext_modules=[
        CUDAExtension('condconv_cuda', [
            'condconv_cuda.cpp','CUDABlas.cpp',
            'CondiConvolutionMM2d.cu',
        ],libraries=["torch_cuda_cu"])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })