# condconv_cuda
cuda实现的条件卷积算子

# 安装
```sh
python setup.py install
```

# 使用示例
```python
import torch
import condconv_cuda 

inp = torch.rand([48,16,128,128]).cuda()
weight = torch.rand([48,5,16,3,3]).cuda()
# bias: Optional[torch.Tensor]=None
bias = torch.rand([48,5]).cuda()

kernel_size =[3,3]
stride = [1,1]
padding = [1,1]
out = CondCONVFunction.apply(inp, weight, kernel_size, bias, stride, padding)
```