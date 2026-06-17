// csrc/pybind.cpp  —  Python binding for the MSAQ-signed kernels.
//
// Registers the four host launchers (defined in the .cu files) under the `msaq`
// namespace via TORCH_LIBRARY, so after `import ms_cuda` they are reachable as
//   torch.ops.msaq.wonly_gemv / wonly_gemm / wa_gemm / kv_decode_attention
// exactly as ms_lib.ops calls them. (TORCH_LIBRARY infers each schema from the
// C++ signature; inference-only, no autograd registered.)

#include <torch/extension.h>

// defined in w_gemv.cu
torch::Tensor wonly_gemv_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper, torch::Tensor shared,
    int64_t OUT, int64_t NB, int64_t u, int64_t gs);
torch::Tensor wonly_gemv_wide_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t OUT, int64_t NB, int64_t gs);

// defined in wa_gemm.cu
torch::Tensor wonly_gemm_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor upper, torch::Tensor shared,
    int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs);
torch::Tensor wa_gemm_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor upper, torch::Tensor shared,
    int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs);

// defined in kv_attention.cu
torch::Tensor kv_decode_attention_cuda(
    torch::Tensor q, torch::Tensor ks, torch::Tensor ku, torch::Tensor kh,
    torch::Tensor vs, torch::Tensor vu, torch::Tensor vh,
    int64_t H, int64_t Lk, int64_t D, int64_t NB, int64_t u, int64_t gs);

// defined in mxint8.cu (baseline)
torch::Tensor mxint8_gemv_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor qweight,
    int64_t OUT, int64_t NB);
torch::Tensor mxint8_gemm_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor qweight,
    int64_t M, int64_t OUT, int64_t K, int64_t NB);
torch::Tensor mxint8_wa_gemm_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor qweight,
    int64_t M, int64_t OUT, int64_t K, int64_t NB);
torch::Tensor mxint8_kv_decode_cuda(
    torch::Tensor q, torch::Tensor ks, torch::Tensor kq,
    torch::Tensor vs, torch::Tensor vq,
    int64_t H, int64_t Lk, int64_t D, int64_t NB);

TORCH_LIBRARY(msaq, m) {
    m.def("wonly_gemv(Tensor x, Tensor scale_exp, Tensor upper, Tensor shared, "
          "int OUT, int NB, int u, int gs) -> Tensor", &wonly_gemv_cuda);
    m.def("wonly_gemv_wide(Tensor x, Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int OUT, int NB, int gs) -> Tensor", &wonly_gemv_wide_cuda);
    m.def("wonly_gemm(Tensor X, Tensor scale_exp, Tensor upper, Tensor shared, "
          "int M, int OUT, int K, int NB, int u, int gs) -> Tensor", &wonly_gemm_cuda);
    m.def("wa_gemm(Tensor X, Tensor scale_exp, Tensor upper, Tensor shared, "
          "int M, int OUT, int K, int NB, int u, int gs) -> Tensor", &wa_gemm_cuda);
    m.def("kv_decode_attention(Tensor q, Tensor ks, Tensor ku, Tensor kh, "
          "Tensor vs, Tensor vu, Tensor vh, "
          "int H, int Lk, int D, int NB, int u, int gs) -> Tensor",
          &kv_decode_attention_cuda);
    // ---- plain MXINT8 baselines ----
    m.def("mxint8_gemv(Tensor x, Tensor scale_exp, Tensor qweight, "
          "int OUT, int NB) -> Tensor", &mxint8_gemv_cuda);
    m.def("mxint8_gemm(Tensor X, Tensor scale_exp, Tensor qweight, "
          "int M, int OUT, int K, int NB) -> Tensor", &mxint8_gemm_cuda);
    m.def("mxint8_wa_gemm(Tensor X, Tensor scale_exp, Tensor qweight, "
          "int M, int OUT, int K, int NB) -> Tensor", &mxint8_wa_gemm_cuda);
    m.def("mxint8_kv_decode(Tensor q, Tensor ks, Tensor kq, "
          "Tensor vs, Tensor vq, int H, int Lk, int D, int NB) -> Tensor",
          &mxint8_kv_decode_cuda);
}
PYBIND11_MODULE(ms_cuda, m) {}
