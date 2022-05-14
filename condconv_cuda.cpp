#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

namespace at
{
  namespace native
  {

    Tensor slow_cond_conv2d_forward_cuda(
        const Tensor &input,
        const Tensor &weight,
        IntArrayRef kernel_size,
        const c10::optional<Tensor> &bias,
        IntArrayRef stride,
        IntArrayRef padding);

    std::tuple<Tensor, Tensor, Tensor> slow_cond_conv2d_backward_cuda(
        const Tensor &grad_output,
        const Tensor &self,
        const Tensor &weight,
        IntArrayRef kernel_size,
        IntArrayRef stride,
        IntArrayRef padding,
        std::array<bool, 3> output_mask);
  } // namespace native

} // namespace at

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward", & at::native::slow_cond_conv2d_forward_cuda, "cond_conv2d_forward (CUDA)");
  m.def("backward", &at::native:: slow_cond_conv2d_backward_cuda, "cond_conv2d_backward (CUDA)");
}