// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// #include <ATen/ATen.h>
#include <ATen/native/cuda/im2col.cuh>
// #include <ATen/cuda/CUDABlas.h>
#include "CUDABlas_cus.h"
#include <ATen/AccumulateType.h>

// #include <ATen/native/ConvUtils.h>
// #include <ATen/Dispatch.h>
// #include <ATen/core/Tensor.h>
// #include <ATen/Functions.h>
// #include <ATen/NativeFunctions.h>

// #define CUDBG 1

#include "accumulate.h"
// #include "MaybeOwned.h"
#include <ATen/div_rtn.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace at { namespace native {
namespace {

bool resize_output_check(const Tensor& output, IntArrayRef shape) {
  // Tests for resizing of tensors with one or more elements
  if (output.sizes().equals(shape)) {
    return false;
  }
  if (output.numel() != 0) {
    TORCH_WARN(
      "An output with one or more elements was resized since it had ",
      "shape ", output.sizes(), ", which does not match the required ",
      "output shape ", shape, ". ",
      "This behavior is deprecated, and in a future PyTorch release outputs ",
      "will not be resized unless they have zero elements. You can explicitly ",
      "reuse an out tensor t by resizing it, inplace, to zero elements with ",
      "t.resize_(0).");
  }
  return true;
}

bool resize_output(const Tensor& output, IntArrayRef shape) {
  if (resize_output_check(output, shape)) {
    // avoid a redispatch for cpu and cuda.
    // TODO: when resize_cuda_ is re-written to be unified with resize_,
    // we can provide the same benefit for cuda.
    //
    // TODO(#61485): functorch wrapped tensors should not go through the
    // fast path. This is a hack, longer term solutions are in the issue

    output.resize_(shape);
    return true;
  } else {
    return false;
  }
}



void slow_cond_conv2d_shape_check(
    const Tensor& input, const Tensor& grad_output,
    const Tensor& weight, const Tensor& bias,
    int64_t kH, int64_t kW,
    int64_t dH, int64_t dW,
    int64_t padH, int64_t padW,
    bool weight_nullable) {
  TORCH_CHECK(kW > 0 && kH > 0,
              "kernel size should be greater than zero, but got kH: ", kH, " kW: ", kW);
  TORCH_CHECK(dW > 0 && dH > 0,
              "stride should be greater than zero, but got dH: ", dH, " dW: ", dW);

  TORCH_CHECK(weight_nullable || weight.defined(),
              "weight tensor is expected to be non-nullable");
  TORCH_CHECK(!weight.defined() ||
              ((weight.numel() > 0) && (weight.dim() == 3)),
              "non-empty 3D weight tensor expected, but got: ", weight.sizes());
  TORCH_CHECK(!bias.defined() || (
                  bias.dim() == 2 
                  && bias.sizes()[0] == weight.sizes()[0]
                  && bias.sizes()[1] == weight.sizes()[1]
              ),
              "Expected bias to have shape [", weight.sizes()[0], "] but got ", bias.sizes());

  const auto in_sizes = input.sizes();// N,dimf(Cin)  ,dimh,dimw
  constexpr int ndim = 4;
  constexpr int dimf = 1;
  constexpr int dimh = 2;
  constexpr int dimw = 3;
  TORCH_CHECK(in_sizes.size() == ndim, "Expected 4D input tensor, but got ", in_sizes);

  // Allow for empty batch size but not other dimensions
  const bool valid_empty = c10::multiply_integers(in_sizes.slice(1)) != 0;
  TORCH_CHECK(valid_empty, "non-empty input tensor expected but got: ", in_sizes);

  int64_t inputHeight = in_sizes[dimh];
  int64_t inputWidth = in_sizes[dimw];

  int64_t exactInputHeight = inputHeight + 2 * padH;
  int64_t exactInputWidth = inputWidth + 2 * padW;

  TORCH_CHECK(exactInputHeight >= kH && exactInputWidth >= kW,
              "Calculated padded input size per channel: ",
              IntArrayRef{exactInputHeight, exactInputWidth},
              ". Kernel size: ", IntArrayRef{kH, kW},
              ". Kernel size can't be greater than actual input size");

  // NOTE: can't use conv_output_size if the weight isn't defined
  auto outputHeight = div_rtn<int64_t>(exactInputHeight - kH, dH) + 1;
  auto outputWidth = div_rtn<int64_t>(exactInputWidth - kW, dW) + 1;

  TORCH_CHECK(outputWidth >= 1 && outputHeight >= 1,
              "Given input size per channel: ",
              IntArrayRef{inputHeight, inputWidth},
              ". Calculated output size per channel: ",
              IntArrayRef{outputHeight, outputWidth},
              ". Output size is too small");

  if (weight.defined()) {
    const auto w_sizes = weight.sizes(); // N,Cout,Cin*h*w
    int64_t nInputPlane = w_sizes[2];
    if (w_sizes.size() == 3) {
      nInputPlane /= (kH * kW);
    }
    TORCH_CHECK(in_sizes[dimf] == nInputPlane,
                "Expected input dim ", dimf, " to have size ", nInputPlane,
                " but got ", in_sizes[dimf]);
  }

  if (grad_output.defined()) {
    const auto gO_sizes = grad_output.sizes();
    TORCH_CHECK(gO_sizes.size() == ndim,
                "Expected grad_output to have ", ndim,
                " dimensions but got shape", gO_sizes);

    if (weight.defined()) {
      const auto w_sizes = weight.sizes();
      TORCH_CHECK(gO_sizes[dimf] == w_sizes[1],//  N,Cout,Cin,h,w
                  "Expected  dim ", dimf, " to have size ", w_sizes[1],
                  " but got ", gO_sizes[dimf]);
    } else if (bias.defined()) {
      const auto b_sizes = bias.sizes();// N,Cout
      // int64_t nOutputPlane = b_sizes.size() == 0 ? 1 : b_sizes[0];
      int64_t nOutputPlane = b_sizes[1];
      TORCH_CHECK(gO_sizes[dimf] == nOutputPlane,
                  "Expected grad_output dim ", dimf, " to have size ",
                  nOutputPlane, " but got ", gO_sizes[dimf]);
    }
    TORCH_CHECK(gO_sizes[dimh] == outputHeight,
                "Expected grad_output dim ", dimh, " to have size ",
                outputHeight, " but got ", gO_sizes[dimh]);
    TORCH_CHECK(gO_sizes[dimw] == outputWidth,
                "Expected grad_output dim ", dimw, " to have size ",
                outputWidth, " but got ", gO_sizes[dimw]);
  }
}

Tensor new_view_weight_MM2d(const Tensor& weight_) {
  auto weight = weight_.contiguous();//  N,Cout,Cin,h,w
  // auto weight = weight_.contiguous();
  const auto w_sizes = weight.sizes();
  TORCH_CHECK(w_sizes.size() == 5);
  int64_t s1 = w_sizes[0];// N
  int64_t s2 = w_sizes[1];// ChannelOut

  int64_t s3 = c10::multiply_integers(w_sizes.slice(2));
  return weight.view({s1, s2 ,s3}); // N , Cout, Cin*w*h
}

void slow_cond_conv2d_forward(
           const Tensor &input,
           const Tensor &output,
           const Tensor &weight_,
           const Tensor &bias,
           int64_t kH, int64_t kW,
           int64_t dH, int64_t dW,
           int64_t padH, int64_t padW) {
  auto weight = new_view_weight_MM2d(weight_); // N , Cout, Cin*w*h
  slow_cond_conv2d_shape_check(
      input, {}, weight, bias, kH, kW, dH, dW, padH, padW, /*weight_nullable*/false);

  constexpr int dimf = 1;
  constexpr int dimh = 2;
  constexpr int dimw = 3;

  auto in_sizes = input.sizes();
  int64_t batchSize = in_sizes[0];
  int64_t nInputPlane  = in_sizes[dimf];
  int64_t inputHeight  = in_sizes[dimh];
  int64_t inputWidth   = in_sizes[dimw];
  int64_t nOutputPlane = weight.sizes()[1];//  N,Cout,Cin,h,w
  int64_t outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
  int64_t outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;

  // Resize output
  resize_output(output, {batchSize, nOutputPlane, outputHeight, outputWidth});
  // 这边有没有问题？

  // Create temporary columns
  auto columns = at::empty({nInputPlane * kW * kH, outputHeight * outputWidth}, input.options());

  const bool requires_columns = (
      kW != 1 || kH != 1 || dW != 1 || dH != 1 || padH != 0 || padW != 0);

  if (bias.defined()) {
    TORCH_CHECK(bias.scalar_type() == input.scalar_type(),
                "Expected bias to have type ", input.scalar_type(),
                " but got ", bias.scalar_type());
#ifdef CUDBG
    std::cout<<"bias"<< bias<<std::endl;
    std::cout<<"output"<< output<<std::endl;
#endif


    output.copy_(bias.view({batchSize,-1, 1, 1}));//加入 batchSize 维度
  } else {
    output.zero_();
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
                                  "slow_cond_conv2d_cuda", [&] {
    // For each elt in batch, do:
    for (int elt = 0; elt < batchSize; elt ++) {
      // Matrix mulitply per output:
      auto input_n = input.select(0, elt);
      auto output_n = output.select(0, elt);
      auto weight_n = weight.select(0, elt);

      if (requires_columns) {
        // Extract columns:
        at::native::im2col(
          c10::cuda::getCurrentCUDAStream(),
          input_n.data_ptr<scalar_t>(),
          nInputPlane, inputHeight, inputWidth,
          outputHeight, outputWidth,
          kH, kW, padH, padW, dH, dW,
          1, 1,
          columns.data_ptr<scalar_t>()
        );
      }

      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      int64_t m = nOutputPlane;
      int64_t n = columns.size(1);
      int64_t k = nInputPlane*kH*kW;

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      auto gemm_in_ptr = requires_columns ?
          columns.data_ptr<scalar_t>() :
          input_n.data_ptr<scalar_t>();
      at::cuda::blas::gemm(
          'n', 'n',
          n, m, k,
          scalar_t(1),
          gemm_in_ptr, n,
          weight_n.data_ptr<scalar_t>(), k,
          scalar_t(1),
          output_n.data_ptr<scalar_t>(), n
      );
    }
  });
}

void slow_cond_conv2d_backward(
    const Tensor &input,
    const Tensor &grad_output,
    const Tensor &grad_input,
    const Tensor &weight_,
    const Tensor &grad_columns,
    int kH, int kW,
    int dH, int dW,
    int padH, int padW) {
  Tensor weight = new_view_weight_MM2d(weight_);
  slow_cond_conv2d_shape_check(input, grad_output, weight, {},
                          kH, kW, dH, dW, padH, padW, /*weight_nullable=*/false);

  // Params
  auto weight_sizes = weight.sizes();// N,Cout,Cin*kh*kw
  int nInputPlane = weight_sizes[2]/(kW*kH);
  int nOutputPlane = weight_sizes[1];

  TORCH_INTERNAL_ASSERT(grad_output.is_contiguous());

  auto input_sizes = input.sizes();
  int64_t inputWidth   = input_sizes[3];
  int64_t inputHeight  = input_sizes[2];
  auto output_sizes = grad_output.sizes();
  int64_t outputWidth  = output_sizes[3];
  int64_t outputHeight = output_sizes[2];

  // Batch size + input planes
  int64_t batchSize = input_sizes[0];

  // Resize output
  resize_output(grad_input, input_sizes);
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");

  // Resize temporary columns
  resize_output(grad_columns, {nInputPlane*kW*kH, outputHeight*outputWidth});
  TORCH_CHECK(grad_columns.is_contiguous(), "grad_columns must be contiguous");

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
                                  "slow_cond_conv2d_backward_cuda", [&] {
    // For each elt in batch, do:
    for (int elt = 0; elt < batchSize; elt ++) {
      // Matrix mulitply per sample:
      auto grad_input_n = grad_input.select(0, elt);
      auto grad_output_n = grad_output.select(0, elt);
      auto weight_n = weight.select(0, elt);

      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      int64_t m = nInputPlane*kW*kH;
      int64_t n = grad_columns.sizes()[1];
      int64_t k = nOutputPlane;

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      at::cuda::blas::gemm<scalar_t>(
          'n', 't',
          n, m, k,
          scalar_t(1),
          grad_output_n.data_ptr<scalar_t>(), n,
          weight_n.data_ptr<scalar_t>(), m,
          scalar_t(0),
          grad_columns.data_ptr<scalar_t>(), n
      );

      // Unpack columns back into input:
      using acc_t = at::acc_type<scalar_t, true>;
      at::native::col2im<scalar_t, acc_t>(
        c10::cuda::getCurrentCUDAStream(),
        grad_columns.data_ptr<scalar_t>(),
        nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW,
        1, 1, grad_input_n.data_ptr<scalar_t>()
      );
    }
  });
}

void slow_cond_conv2d_grad_weight(
           const Tensor &input,
           const Tensor &grad_output,
           const Tensor &grad_weight_,
           const Tensor &columns,
           int64_t kH, int64_t kW,
           int64_t dH, int64_t dW,
           int64_t padH, int64_t padW) {
  TORCH_CHECK(grad_weight_.is_contiguous(), "grad_weight needs to be contiguous");
  auto grad_weight = new_view_weight_MM2d(grad_weight_);
  slow_cond_conv2d_shape_check(input, grad_output, grad_weight, {},
                          kH, kW, dH, dW, padH, padW, /*weight_nullable=*/true);

  // Params
  TORCH_INTERNAL_ASSERT(input.is_contiguous());
  TORCH_INTERNAL_ASSERT(grad_output.is_contiguous());

  auto input_sizes = input.sizes();
  int64_t nInputPlane = input_sizes[1];
  int64_t nOutputPlane = grad_output.sizes()[1];

  int64_t inputWidth   = input_sizes[3];
  int64_t inputHeight  = input_sizes[2];
  int64_t outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  int64_t outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  // Batch size + input planes
  int64_t batchSize = input_sizes[0];

  // Resize temporary columns
  resize_output(columns, {nInputPlane * kH * kW, outputHeight * outputWidth});

  const bool requires_columns = (
      kW != 1 || kH != 1 || dW != 1 || dH != 1 || padH != 0 || padW != 0);

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
                                  "slow_cond_conv2d_grad_weight_cuda", [&] {
    // For each elt in batch, do:
    for (int elt = 0; elt < batchSize; elt ++) {
      // Matrix mulitply per output:
      auto grad_output_n = grad_output.select(0, elt);
      auto grad_weight_n = grad_weight.select(0, elt);

      // Matrix mulitply per output:
      auto input_n = input.select(0, elt);

      if (requires_columns) {
        // Extract columns:
        at::native::im2col<scalar_t>(
          c10::cuda::getCurrentCUDAStream(),
          input_n.data_ptr<scalar_t>(),
          nInputPlane, inputHeight, inputWidth,
          outputHeight, outputWidth,
          kH, kW, padH, padW, dH, dW,
          1, 1,
          columns.data_ptr<scalar_t>()
        );
      }

      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      int64_t m = nOutputPlane;
      int64_t n = nInputPlane*kW*kH;
      int64_t k = columns.sizes()[1];

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      auto gemm_in_ptr = requires_columns ?
          columns.data_ptr<scalar_t>() :
          input_n.data_ptr<scalar_t>();
      at::cuda::blas::gemm(
          't', 'n',
          n, m, k,
          scalar_t(1),
          gemm_in_ptr, k,
          grad_output_n.data_ptr<scalar_t>(), k,
          scalar_t(1),
          grad_weight_n.data_ptr<scalar_t>(), n
      );
    }
  });
}

}  // namespace (anonymous)


Tensor& slow_cond_conv2d_forward_out_cuda(
    const Tensor &self_,
    const Tensor &weight_,
    IntArrayRef kernel_size,
    const c10::optional<Tensor> &bias_,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor &output) {
  TORCH_CHECK(kernel_size.size() == 2);
  TORCH_CHECK(stride.size() == 2);
  TORCH_CHECK(padding.size() == 2);
  Tensor undefined;

  const Tensor self = self_.contiguous();
  const Tensor weight = weight_.contiguous();
  // auto bias = [&] {
  //   if (bias_.has_value() && bias_->defined()) {
  //     return expect_contiguous(*bias_);
  //   }
  //   return MaybeOwned<Tensor>::owned(c10::in_place);
  // }();
  const Tensor bias = (bias_->defined() ? bias_->contiguous() : undefined);

  slow_cond_conv2d_forward(
      self,
      output,
      weight,
      bias,
      kernel_size[0], kernel_size[1],
      stride[0], stride[1],
      padding[0], padding[1]
    );
  return output;
}

Tensor slow_cond_conv2d_forward_cuda(
    const Tensor &self,
    const Tensor &weight,
    IntArrayRef kernel_size,
    const c10::optional<Tensor> &bias,
    IntArrayRef stride,
    IntArrayRef padding) {
  auto output = at::empty({0}, self.options());
  return slow_cond_conv2d_forward_out_cuda(
      self, weight, kernel_size, bias, stride, padding, output);
}

std::tuple<Tensor&, Tensor&, Tensor&> slow_cond_conv2d_backward_out_cuda(
    const Tensor& grad_output_,
    const Tensor& self_,
    const Tensor& weight_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias) {

  auto grad_output = grad_output_.contiguous();

  Tensor columns = at::empty({0}, self_.options());
  if (grad_input.defined()) {
    resize_output(grad_input, self_.sizes());
    
    auto weight = weight_.contiguous();

    slow_cond_conv2d_backward(// grad_input 是本函数的output
        self_, grad_output,
        grad_input, weight,
        columns,
        kernel_size[0], kernel_size[1],
        stride[0], stride[1],
        padding[0], padding[1]);
  }
  if (grad_bias.defined()) {
    at::sum_out(grad_bias, grad_output, IntArrayRef{ 2, 3});  // 0,2,3 ->2,3 保留 batch 维度
  }
  if (grad_weight.defined()) {
    resize_output(grad_weight, weight_.sizes());
    grad_weight.zero_();
    auto self = self_.contiguous();

    slow_cond_conv2d_grad_weight(
        self,
        grad_output,
        grad_weight,  //output
        columns,
        kernel_size[0], kernel_size[1],
        stride[0], stride[1],
        padding[0], padding[1]
      );
  }
  return std::tuple<Tensor&, Tensor&, Tensor&>{
      grad_input, grad_weight, grad_bias};
}

std::tuple<Tensor, Tensor, Tensor> slow_cond_conv2d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    std::array<bool, 3> output_mask) {
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;

  if (output_mask[0]) {
    grad_input = at::empty({0}, grad_output.options());
  }

  if (output_mask[1]) {
    grad_weight = at::empty({0}, grad_output.options());
  }

  if (output_mask[2]) {
    grad_bias = at::empty({0}, grad_output.options());
  }

  return slow_cond_conv2d_backward_out_cuda(
      grad_output,
      self,
      weight,
      kernel_size,
      stride,
      padding,
      grad_input,
      grad_weight,
      grad_bias);
}

} // namespace native
} // namespace at