# condconv_cuda
cuda实现的条件卷积算子

# 依赖
pytorch >= 1.8.1
cuda >= 11.1

# 安装
```sh
python setup.py install
```

# 使用示例
```python

# 建议通过 condconvFun 来使用，已经封装为可自动求导函数
import torch
from . import condconvFun  # 请自行解决路径问题

inp = torch.rand([48,16,128,128]).cuda()
weight = torch.rand([48,5,16,3,3]).cuda()
# bias: Optional[torch.Tensor]=None
bias = torch.rand([48,5]).cuda()

kernel_size =[3,3]
stride = [1,1]
padding = [1,1]
out = condconvFun.CondCONVFunction.apply(inp, weight, kernel_size, bias, stride, padding)
```