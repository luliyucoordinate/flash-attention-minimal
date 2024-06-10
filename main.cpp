#include <torch/extension.h>

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor forward_v2(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", torch::wrap_pybind_function(forward), "forward");
  m.def("forward_v2", torch::wrap_pybind_function(forward_v2), "forward_v2");
}