from typing import Optional
import torch
import condconv_cuda

class CondCONVFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, kernel_size, bias: Optional[torch.Tensor] , stride, padding):

        # const Tensor &input,
        # const Tensor &weight,
        # IntArrayRef kernel_size,
        # const c10::optional<Tensor> &bias,
        # IntArrayRef stride,
        # IntArrayRef padding);
    
        output = condconv_cuda.forward(input,weight, kernel_size, bias, stride, padding)
        output_mask = [True,True,bias is not None]

        ctx.save_for_backward(input,weight,bias)

        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_mask = output_mask
        return output

    @staticmethod
    def backward(ctx, grad_out):


    # const Tensor& grad_output,
    # const Tensor& self,
    # const Tensor& weight,
    # IntArrayRef kernel_size,
    # IntArrayRef stride,
    # IntArrayRef padding,
    # std::array<bool, 3> output_mask) 

        outputs = condconv_cuda.backward(grad_out,ctx.saved_tensors[0],ctx.saved_tensors[1],
                ctx.kernel_size,  ctx.stride, ctx.padding,  ctx.output_mask)

        grad_input, grad_weight, grad_bias = outputs

        return grad_input, grad_weight,None, grad_bias,None,None # backward的输出应该对应 forward 输入




if __name__=="__main__":
    import os
    import torch.nn.functional as F
    F.conv1d

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    # help(CondCONVFunction)
    inp = torch.rand([48,16,128,128]).clone().detach().cuda()
    weight = torch.rand([48,5,16,3,3]).clone().detach().requires_grad_(True) .cuda()
    # bias: Optional[torch.Tensor]=None
    bias = torch.rand([48,5]).clone().detach().cuda()

    target = torch.Tensor( torch.ones([48,5,128,128])).cuda()
    

    kernel_size =[3,3]
    stride = [1,1]
    padding = [1,1]
    out = CondCONVFunction.apply(inp, weight, kernel_size, bias, stride, padding)

    loss = F.mse_loss(out,target)
    loss.backward()


    inp1d = torch.ones([48,16,128,128]).clone().detach().requires_grad_(True).cuda()
    weight1d = torch.ones([5,16,3,3]).clone().detach().requires_grad_(True).cuda()
    bias_conv1d = torch.ones([5]).clone().detach().requires_grad_(True).cuda()
    out1d = F.conv2d(inp1d,weight1d,bias_conv1d,padding=1)
    loss2 = F.mse_loss(out1d,target)
    loss2.backward()

    pass