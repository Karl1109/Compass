#include <torch/extension.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor wkv_cuda_forward(
    torch::Tensor w, 
    torch::Tensor u, 
    torch::Tensor k, 
    torch::Tensor v);

std::vector<torch::Tensor> wkv_cuda_backward(
    torch::Tensor w, 
    torch::Tensor u, 
    torch::Tensor k, 
    torch::Tensor v, 
    torch::Tensor gy);

torch::Tensor wkv_forward(
    torch::Tensor w, 
    torch::Tensor u, 
    torch::Tensor k, 
    torch::Tensor v) {
    CHECK_INPUT(w);
    CHECK_INPUT(u);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    return wkv_cuda_forward(w, u, k, v);
}

std::vector<torch::Tensor> wkv_backward(
    torch::Tensor w, 
    torch::Tensor u, 
    torch::Tensor k, 
    torch::Tensor v, 
    torch::Tensor gy) {
    CHECK_INPUT(w);
    CHECK_INPUT(u);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(gy);
    return wkv_cuda_backward(w, u, k, v, gy);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wkv_forward", &wkv_forward, "WKV Forward(CUDA)");
    m.def("wkv_backward", &wkv_backward, "WKV Backward(CUDA)");
}

TORCH_LIBRARY(wkv, m) {
    m.def("wkv_forward", wkv_forward);
    m.def("wkv_backward", wkv_backward);
}