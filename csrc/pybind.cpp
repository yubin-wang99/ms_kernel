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
    int64_t OUT, int64_t NB, int64_t u, int64_t gs);
torch::Tensor wa_gemv_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t OUT, int64_t NB, int64_t u, int64_t gs);
torch::Tensor wonly_gemv_batched_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t M, int64_t OUT, int64_t NB, int64_t u, int64_t gs);
torch::Tensor wonly_gemv_batched_relayout_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor hi4_cm, torch::Tensor lowun_cm,
    torch::Tensor shared_cm, int64_t M, int64_t OUT, int64_t NB, int64_t u, int64_t gs);
torch::Tensor wonly_gemv_batched_unsigned_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t M, int64_t OUT, int64_t NB, int64_t u, int64_t gs);
torch::Tensor wonly_gemv_batched_ra_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper_ra_cm, torch::Tensor shared_cm,
    int64_t M, int64_t OUT, int64_t NB, int64_t u, int64_t gs);
torch::Tensor wonly_gemv_batched_ra_sepsc_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper_ra_cm, torch::Tensor shared_cm,
    int64_t M, int64_t OUT, int64_t NB, int64_t u, int64_t gs);
torch::Tensor wonly_gemv_batched_densebfe_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t M, int64_t OUT, int64_t NB, int64_t u, int64_t gs);
torch::Tensor ms_dequant_bf16_unsigned_cuda(
    torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs);
torch::Tensor wonly_gemv_wide_unsigned_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t OUT, int64_t NB, int64_t u, int64_t gs);
torch::Tensor wa_gemv_unsigned_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t OUT, int64_t NB, int64_t u, int64_t gs);
torch::Tensor wa_gemv_batched_unsigned_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t M, int64_t OUT, int64_t NB, int64_t u, int64_t gs);
torch::Tensor kv_kdot_unsigned_cuda(
    torch::Tensor q, torch::Tensor ks, torch::Tensor ku, torch::Tensor kh,
    int64_t B, int64_t H, int64_t Hkv, int64_t Lk, int64_t D, int64_t NB, int64_t u, int64_t gs);
torch::Tensor kv_kdot_mxint8_cuda(
    torch::Tensor q, torch::Tensor ks, torch::Tensor kq,
    int64_t B, int64_t H, int64_t Hkv, int64_t Lk, int64_t D, int64_t NB);
torch::Tensor wonly_gemv_tc_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper, torch::Tensor shared,
    int64_t M, int64_t OUT, int64_t NB, int64_t u, int64_t gs);
torch::Tensor wa_gemv_batched_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t M, int64_t OUT, int64_t NB, int64_t u, int64_t gs);

// defined in wa_gemm.cu
std::vector<torch::Tensor> quant_act_unsigned_cuda(torch::Tensor X, int64_t M, int64_t K,
                                                   int64_t NB, int64_t u, int64_t gs);
std::vector<torch::Tensor> quant_act_cuda(torch::Tensor X, int64_t M, int64_t K,
                                          int64_t NB, int64_t u, int64_t gs);
torch::Tensor wonly_gemm_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor upper, torch::Tensor shared,
    int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs);
torch::Tensor wonly_gemm_cm_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs);
torch::Tensor ms_dequant_bf16_cuda(
    torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs);
torch::Tensor msfp8_dequant_bf16_cuda(
    torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs);
torch::Tensor msfp8_gemv_wide_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t OUT, int64_t NB, int64_t u, int64_t gs);
torch::Tensor msfp8_gemv_batched_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t M, int64_t OUT, int64_t NB, int64_t u, int64_t gs);
torch::Tensor msfp8_gemm_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs);
torch::Tensor wonly_gemm_tc_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs);
torch::Tensor wonly_gemm_fused_skinny_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor upper, torch::Tensor shared,
    int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs);
torch::Tensor mxint8_gemm_fused_skinny_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor qweight_cm,
    int64_t M, int64_t OUT, int64_t K, int64_t NB);
torch::Tensor wa_gemm_fused_imma_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs);
torch::Tensor wa_gemm_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor upper, torch::Tensor shared,
    int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs);
torch::Tensor wa_gemm_cm_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs);

// defined in kv_attention.cu
torch::Tensor kv_decode_attention_cuda(
    torch::Tensor q, torch::Tensor ks, torch::Tensor ku, torch::Tensor kh,
    torch::Tensor vs, torch::Tensor vu, torch::Tensor vh,
    int64_t H, int64_t Hkv, int64_t Lk, int64_t D, int64_t NB, int64_t u, int64_t gs,
    int64_t Lcap);
torch::Tensor kv_decode_attention_batched_cuda(
    torch::Tensor q, torch::Tensor ks, torch::Tensor ku, torch::Tensor kh,
    torch::Tensor vs, torch::Tensor vu, torch::Tensor vh,
    int64_t B, int64_t H, int64_t Hkv, int64_t Lk, int64_t D, int64_t NB, int64_t u, int64_t gs,
    int64_t Lcap);
torch::Tensor kv_kdot_uspec_cuda(
    torch::Tensor q, torch::Tensor ks, torch::Tensor ku, torch::Tensor kh,
    int64_t B, int64_t H, int64_t Hkv, int64_t Lk, int64_t D, int64_t NB, int64_t u, int64_t gs);
torch::Tensor kv_kdot_relayout_cuda(
    torch::Tensor q, torch::Tensor ks, torch::Tensor hi4, torch::Tensor lowun, torch::Tensor shared,
    int64_t B, int64_t H, int64_t Hkv, int64_t Lk, int64_t D, int64_t NB, int64_t u, int64_t gs);
torch::Tensor pv_wmma_cuda(
    torch::Tensor P, torch::Tensor vs, torch::Tensor vu, torch::Tensor vh,
    int64_t Hkv, int64_t M, int64_t D, int64_t Lk, int64_t NBd, int64_t u, int64_t gs);
torch::Tensor pv_wmma_mx_cuda(
    torch::Tensor P, torch::Tensor vs, torch::Tensor vq,
    int64_t Hkv, int64_t M, int64_t D, int64_t Lk, int64_t NBd);
torch::Tensor qk_wmma_cuda(
    torch::Tensor Q, torch::Tensor ks, torch::Tensor ku, torch::Tensor kh,
    int64_t Hkv, int64_t M, int64_t D, int64_t Lk, int64_t NBd, int64_t u, int64_t gs);
torch::Tensor qk_wmma_mx_cuda(
    torch::Tensor Q, torch::Tensor ks, torch::Tensor kq,
    int64_t Hkv, int64_t M, int64_t D, int64_t Lk, int64_t NBd);
std::vector<torch::Tensor> kv_write_cuda(
    torch::Tensor X, int64_t H, int64_t L, int64_t D, int64_t NB, int64_t u, int64_t gs);
void kv_append_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor upper, torch::Tensor shared,
    int64_t H, int64_t D, int64_t NB, int64_t pos, int64_t Lcap, int64_t u, int64_t gs);
void kv_append_rot_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor upper, torch::Tensor shared,
    int64_t H, int64_t D, int64_t NB, int64_t pos, int64_t Lcap, int64_t u, int64_t gs);
// MXFP8-MSAQ (E3M4) KV — 1:1 analogs of the INT MSAQ KV ops
torch::Tensor msfp8_kv_decode_attention_cuda(
    torch::Tensor q, torch::Tensor ks, torch::Tensor ku, torch::Tensor kh,
    torch::Tensor vs, torch::Tensor vu, torch::Tensor vh,
    int64_t H, int64_t Hkv, int64_t Lk, int64_t D, int64_t NB, int64_t u, int64_t gs, int64_t Lcap);
torch::Tensor msfp8_kv_decode_attention_batched_cuda(
    torch::Tensor q, torch::Tensor ks, torch::Tensor ku, torch::Tensor kh,
    torch::Tensor vs, torch::Tensor vu, torch::Tensor vh,
    int64_t B, int64_t H, int64_t Hkv, int64_t Lk, int64_t D, int64_t NB, int64_t u, int64_t gs, int64_t Lcap);
torch::Tensor msfp8_kv_kdot_cuda(
    torch::Tensor q, torch::Tensor ks, torch::Tensor ku, torch::Tensor kh,
    int64_t B, int64_t H, int64_t Hkv, int64_t Lk, int64_t D, int64_t NB, int64_t u, int64_t gs);
std::vector<torch::Tensor> msfp8_kv_write_cuda(
    torch::Tensor X, int64_t H, int64_t L, int64_t D, int64_t NB, int64_t u, int64_t gs);
void msfp8_kv_append_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor upper, torch::Tensor shared,
    int64_t H, int64_t D, int64_t NB, int64_t pos, int64_t Lcap, int64_t u, int64_t gs);

// defined in rotate.cu
torch::Tensor hadamard_rotate_cuda(torch::Tensor x);

// defined in mxint8.cu (baseline)
torch::Tensor mxint8_gemv_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor qweight,
    int64_t OUT, int64_t NB);
// column-major wide-load variants (matched to the MSAQ wide-load kernels)
torch::Tensor mxint8_dequant_bf16_cuda(
    torch::Tensor scale_exp, torch::Tensor qweight_cm, int64_t OUT, int64_t K, int64_t NB);
torch::Tensor mxint8_gemv_wide_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor qweight_cm, int64_t OUT, int64_t NB);
torch::Tensor mxint8_gemv_batched_wide_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor qweight_cm, int64_t M, int64_t OUT, int64_t NB);
torch::Tensor mxint8_wa_gemv_wide_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor qweight_cm, int64_t OUT, int64_t NB);
torch::Tensor mxint8_wa_gemv_batched_wide_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor qweight_cm, int64_t M, int64_t OUT, int64_t NB);
torch::Tensor mxint8_gemm_cm_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor qweight_cm, int64_t M, int64_t OUT, int64_t K, int64_t NB);
torch::Tensor mxint8_wa_gemm_cm_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor qweight_cm, int64_t M, int64_t OUT, int64_t K, int64_t NB);
torch::Tensor mxint8_gemv_batched_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor qweight,
    int64_t M, int64_t OUT, int64_t NB);
torch::Tensor mxint8_wa_gemv_batched_cuda(
    torch::Tensor x, torch::Tensor scale_exp, torch::Tensor qweight,
    int64_t M, int64_t OUT, int64_t NB);
torch::Tensor mxint8_wa_gemv_cuda(
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
    int64_t H, int64_t Hkv, int64_t Lk, int64_t D, int64_t NB, int64_t Lcap);
torch::Tensor mxint8_kv_decode_batched_cuda(
    torch::Tensor q, torch::Tensor ks, torch::Tensor kq,
    torch::Tensor vs, torch::Tensor vq,
    int64_t B, int64_t H, int64_t Hkv, int64_t Lk, int64_t D, int64_t NB, int64_t Lcap);
std::vector<torch::Tensor> mxint8_kv_write_cuda(
    torch::Tensor X, int64_t H, int64_t L, int64_t D, int64_t NB);
void mxint8_kv_append_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor qweight,
    int64_t H, int64_t D, int64_t NB, int64_t pos, int64_t Lcap);

TORCH_LIBRARY(msaq, m) {
    m.def("wonly_gemv(Tensor x, Tensor scale_exp, Tensor upper, Tensor shared, "
          "int OUT, int NB, int u, int gs) -> Tensor", &wonly_gemv_cuda);
    m.def("wonly_gemv_wide(Tensor x, Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int OUT, int NB, int u, int gs) -> Tensor", &wonly_gemv_wide_cuda);
    m.def("wa_gemv(Tensor x, Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int OUT, int NB, int u, int gs) -> Tensor", &wa_gemv_cuda);
    m.def("wonly_gemv_batched(Tensor x, Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int M, int OUT, int NB, int u, int gs) -> Tensor", &wonly_gemv_batched_cuda);
    m.def("wonly_gemv_batched_relayout(Tensor x, Tensor scale_exp, Tensor hi4_cm, Tensor lowun_cm, "
          "Tensor shared_cm, int M, int OUT, int NB, int u, int gs) -> Tensor", &wonly_gemv_batched_relayout_cuda);
    m.def("wonly_gemv_batched_unsigned(Tensor x, Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int M, int OUT, int NB, int u, int gs) -> Tensor", &wonly_gemv_batched_unsigned_cuda);
    m.def("wonly_gemv_batched_ra(Tensor x, Tensor scale_exp, Tensor upper_ra_cm, Tensor shared_cm, "
          "int M, int OUT, int NB, int u, int gs) -> Tensor", &wonly_gemv_batched_ra_cuda);
    m.def("wonly_gemv_batched_ra_sepsc(Tensor x, Tensor scale_exp, Tensor upper_ra_cm, Tensor shared_cm, "
          "int M, int OUT, int NB, int u, int gs) -> Tensor", &wonly_gemv_batched_ra_sepsc_cuda);
    m.def("wonly_gemv_batched_densebfe(Tensor x, Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int M, int OUT, int NB, int u, int gs) -> Tensor", &wonly_gemv_batched_densebfe_cuda);
    m.def("ms_dequant_bf16_unsigned(Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int OUT, int K, int NB, int u, int gs) -> Tensor", &ms_dequant_bf16_unsigned_cuda);
    m.def("wonly_gemv_wide_unsigned(Tensor x, Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int OUT, int NB, int u, int gs) -> Tensor", &wonly_gemv_wide_unsigned_cuda);
    m.def("wa_gemv_unsigned(Tensor x, Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int OUT, int NB, int u, int gs) -> Tensor", &wa_gemv_unsigned_cuda);
    m.def("wa_gemv_batched_unsigned(Tensor x, Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int M, int OUT, int NB, int u, int gs) -> Tensor", &wa_gemv_batched_unsigned_cuda);
    m.def("kv_kdot_unsigned(Tensor q, Tensor ks, Tensor ku, Tensor kh, int B, int H, int Hkv, "
          "int Lk, int D, int NB, int u, int gs) -> Tensor", &kv_kdot_unsigned_cuda);
    m.def("kv_kdot_mxint8(Tensor q, Tensor ks, Tensor kq, int B, int H, int Hkv, "
          "int Lk, int D, int NB) -> Tensor", &kv_kdot_mxint8_cuda);
    m.def("wonly_gemv_tc(Tensor x, Tensor scale_exp, Tensor upper, Tensor shared, "
          "int M, int OUT, int NB, int u, int gs) -> Tensor", &wonly_gemv_tc_cuda);
    m.def("wa_gemv_batched(Tensor x, Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int M, int OUT, int NB, int u, int gs) -> Tensor", &wa_gemv_batched_cuda);
    m.def("quant_act(Tensor X, int M, int K, int NB, int u, int gs) -> Tensor[]", &quant_act_cuda);
    m.def("quant_act_unsigned(Tensor X, int M, int K, int NB, int u, int gs) -> Tensor[]", &quant_act_unsigned_cuda);
    m.def("kv_write(Tensor X, int H, int L, int D, int NB, int u, int gs) -> Tensor[]", &kv_write_cuda);
    m.def("kv_append(Tensor X, Tensor(a!) scale_exp, Tensor(b!) upper, Tensor(c!) shared, "
          "int H, int D, int NB, int pos, int Lcap, int u, int gs) -> ()", &kv_append_cuda);
    m.def("kv_append_rot(Tensor X, Tensor(a!) scale_exp, Tensor(b!) upper, Tensor(c!) shared, "
          "int H, int D, int NB, int pos, int Lcap, int u, int gs) -> ()", &kv_append_rot_cuda);
    m.def("wonly_gemm(Tensor X, Tensor scale_exp, Tensor upper, Tensor shared, "
          "int M, int OUT, int K, int NB, int u, int gs) -> Tensor", &wonly_gemm_cuda);
    m.def("wonly_gemm_cm(Tensor X, Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int M, int OUT, int K, int NB, int u, int gs) -> Tensor", &wonly_gemm_cm_cuda);
    m.def("ms_dequant_bf16(Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int OUT, int K, int NB, int u, int gs) -> Tensor", &ms_dequant_bf16_cuda);
    m.def("msfp8_dequant_bf16(Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int OUT, int K, int NB, int u, int gs) -> Tensor", &msfp8_dequant_bf16_cuda);
    m.def("msfp8_gemv_wide(Tensor x, Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int OUT, int NB, int u, int gs) -> Tensor", &msfp8_gemv_wide_cuda);
    m.def("msfp8_gemv_batched(Tensor x, Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int M, int OUT, int NB, int u, int gs) -> Tensor", &msfp8_gemv_batched_cuda);
    m.def("msfp8_gemm(Tensor X, Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int M, int OUT, int K, int NB, int u, int gs) -> Tensor", &msfp8_gemm_cuda);
    m.def("wonly_gemm_tc(Tensor X, Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int M, int OUT, int K, int NB, int u, int gs) -> Tensor", &wonly_gemm_tc_cuda);
    m.def("wonly_gemm_fused_skinny(Tensor X, Tensor scale_exp, Tensor upper, Tensor shared, "
          "int M, int OUT, int K, int NB, int u, int gs) -> Tensor", &wonly_gemm_fused_skinny_cuda);
    m.def("mxint8_gemm_fused_skinny(Tensor X, Tensor scale_exp, Tensor qweight_cm, "
          "int M, int OUT, int K, int NB) -> Tensor", &mxint8_gemm_fused_skinny_cuda);
    m.def("wa_gemm_fused_imma(Tensor X, Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int M, int OUT, int K, int NB, int u, int gs) -> Tensor", &wa_gemm_fused_imma_cuda);
    m.def("wa_gemm(Tensor X, Tensor scale_exp, Tensor upper, Tensor shared, "
          "int M, int OUT, int K, int NB, int u, int gs) -> Tensor", &wa_gemm_cuda);
    m.def("wa_gemm_cm(Tensor X, Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int M, int OUT, int K, int NB, int u, int gs) -> Tensor", &wa_gemm_cm_cuda);
    m.def("kv_decode_attention(Tensor q, Tensor ks, Tensor ku, Tensor kh, "
          "Tensor vs, Tensor vu, Tensor vh, "
          "int H, int Hkv, int Lk, int D, int NB, int u, int gs, int Lcap=-1) -> Tensor",
          &kv_decode_attention_cuda);
    m.def("kv_decode_attention_batched(Tensor q, Tensor ks, Tensor ku, Tensor kh, "
          "Tensor vs, Tensor vu, Tensor vh, "
          "int B, int H, int Hkv, int Lk, int D, int NB, int u, int gs, int Lcap=-1) -> Tensor",
          &kv_decode_attention_batched_cuda);
    m.def("kv_kdot_uspec(Tensor q, Tensor ks, Tensor ku, Tensor kh, int B, int H, int Hkv, "
          "int Lk, int D, int NB, int u, int gs) -> Tensor", &kv_kdot_uspec_cuda);
    m.def("msfp8_kv_decode_attention(Tensor q, Tensor ks, Tensor ku, Tensor kh, "
          "Tensor vs, Tensor vu, Tensor vh, "
          "int H, int Hkv, int Lk, int D, int NB, int u, int gs, int Lcap=-1) -> Tensor",
          &msfp8_kv_decode_attention_cuda);
    m.def("msfp8_kv_decode_attention_batched(Tensor q, Tensor ks, Tensor ku, Tensor kh, "
          "Tensor vs, Tensor vu, Tensor vh, "
          "int B, int H, int Hkv, int Lk, int D, int NB, int u, int gs, int Lcap=-1) -> Tensor",
          &msfp8_kv_decode_attention_batched_cuda);
    m.def("msfp8_kv_kdot(Tensor q, Tensor ks, Tensor ku, Tensor kh, int B, int H, int Hkv, "
          "int Lk, int D, int NB, int u, int gs) -> Tensor", &msfp8_kv_kdot_cuda);
    m.def("msfp8_kv_write(Tensor X, int H, int L, int D, int NB, int u, int gs) -> Tensor[]",
          &msfp8_kv_write_cuda);
    m.def("msfp8_kv_append(Tensor X, Tensor(a!) scale_exp, Tensor(b!) upper, Tensor(c!) shared, "
          "int H, int D, int NB, int pos, int Lcap, int u, int gs) -> ()", &msfp8_kv_append_cuda);
    m.def("kv_kdot_relayout(Tensor q, Tensor ks, Tensor hi4, Tensor lowun, Tensor shared, "
          "int B, int H, int Hkv, int Lk, int D, int NB, int u, int gs) -> Tensor", &kv_kdot_relayout_cuda);
    m.def("pv_wmma(Tensor P, Tensor vs, Tensor vu, Tensor vh, "
          "int Hkv, int M, int D, int Lk, int NBd, int u, int gs) -> Tensor", &pv_wmma_cuda);
    m.def("pv_wmma_mx(Tensor P, Tensor vs, Tensor vq, "
          "int Hkv, int M, int D, int Lk, int NBd) -> Tensor", &pv_wmma_mx_cuda);
    m.def("qk_wmma(Tensor Q, Tensor ks, Tensor ku, Tensor kh, "
          "int Hkv, int M, int D, int Lk, int NBd, int u, int gs) -> Tensor", &qk_wmma_cuda);
    m.def("qk_wmma_mx(Tensor Q, Tensor ks, Tensor kq, "
          "int Hkv, int M, int D, int Lk, int NBd) -> Tensor", &qk_wmma_mx_cuda);
    m.def("hadamard_rotate(Tensor x) -> Tensor", &hadamard_rotate_cuda);
    // ---- plain MXINT8 baselines ----
    m.def("mxint8_dequant_bf16(Tensor scale_exp, Tensor qweight_cm, "
          "int OUT, int K, int NB) -> Tensor", &mxint8_dequant_bf16_cuda);
    m.def("mxint8_gemv_wide(Tensor x, Tensor scale_exp, Tensor qweight_cm, "
          "int OUT, int NB) -> Tensor", &mxint8_gemv_wide_cuda);
    m.def("mxint8_gemv_batched_wide(Tensor x, Tensor scale_exp, Tensor qweight_cm, "
          "int M, int OUT, int NB) -> Tensor", &mxint8_gemv_batched_wide_cuda);
    m.def("mxint8_wa_gemv_wide(Tensor x, Tensor scale_exp, Tensor qweight_cm, "
          "int OUT, int NB) -> Tensor", &mxint8_wa_gemv_wide_cuda);
    m.def("mxint8_wa_gemv_batched_wide(Tensor x, Tensor scale_exp, Tensor qweight_cm, "
          "int M, int OUT, int NB) -> Tensor", &mxint8_wa_gemv_batched_wide_cuda);
    m.def("mxint8_gemm_cm(Tensor X, Tensor scale_exp, Tensor qweight_cm, "
          "int M, int OUT, int K, int NB) -> Tensor", &mxint8_gemm_cm_cuda);
    m.def("mxint8_wa_gemm_cm(Tensor X, Tensor scale_exp, Tensor qweight_cm, "
          "int M, int OUT, int K, int NB) -> Tensor", &mxint8_wa_gemm_cm_cuda);
    m.def("mxint8_gemv(Tensor x, Tensor scale_exp, Tensor qweight, "
          "int OUT, int NB) -> Tensor", &mxint8_gemv_cuda);
    m.def("mxint8_gemv_batched(Tensor x, Tensor scale_exp, Tensor qweight, "
          "int M, int OUT, int NB) -> Tensor", &mxint8_gemv_batched_cuda);
    m.def("mxint8_wa_gemv_batched(Tensor x, Tensor scale_exp, Tensor qweight, "
          "int M, int OUT, int NB) -> Tensor", &mxint8_wa_gemv_batched_cuda);
    m.def("mxint8_wa_gemv(Tensor x, Tensor scale_exp, Tensor qweight, "
          "int OUT, int NB) -> Tensor", &mxint8_wa_gemv_cuda);
    m.def("mxint8_gemm(Tensor X, Tensor scale_exp, Tensor qweight, "
          "int M, int OUT, int K, int NB) -> Tensor", &mxint8_gemm_cuda);
    m.def("mxint8_wa_gemm(Tensor X, Tensor scale_exp, Tensor qweight, "
          "int M, int OUT, int K, int NB) -> Tensor", &mxint8_wa_gemm_cuda);
    m.def("mxint8_kv_decode(Tensor q, Tensor ks, Tensor kq, "
          "Tensor vs, Tensor vq, int H, int Hkv, int Lk, int D, int NB, int Lcap=-1) -> Tensor",
          &mxint8_kv_decode_cuda);
    m.def("mxint8_kv_decode_batched(Tensor q, Tensor ks, Tensor kq, Tensor vs, Tensor vq, "
          "int B, int H, int Hkv, int Lk, int D, int NB, int Lcap=-1) -> Tensor",
          &mxint8_kv_decode_batched_cuda);
    m.def("mxint8_kv_write(Tensor X, int H, int L, int D, int NB) -> Tensor[]", &mxint8_kv_write_cuda);
    m.def("mxint8_kv_append(Tensor X, Tensor(a!) scale_exp, Tensor(b!) qweight, "
          "int H, int D, int NB, int pos, int Lcap) -> ()", &mxint8_kv_append_cuda);
}
PYBIND11_MODULE(ms_cuda, m) {}
