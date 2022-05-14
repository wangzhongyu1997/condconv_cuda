from torch.utils.cpp_extension import load
cd_conv = load(name="cd_conv",sources=["condconv_cuda.cpp","CondiConvolutionMM2d.cu","CUDABlas.cpp"])
# 本方式在运行时编译c++代码，你可以忽略
pass