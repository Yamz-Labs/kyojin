#pragma once
#if defined(USE_ROCM)
#include <ATen/Tensor.h>
#include <vector>
void gm_moe_kq_decode(const at::Tensor& h, const at::Tensor& sel, const at::Tensor& wts, const at::Tensor& gate_ptrs,
                      const at::Tensor& up_ptrs, const at::Tensor& down_ptrs, int64_t gate_stride, int64_t up_stride,
                      int64_t down_stride, int64_t fmt_d, at::Tensor& act, at::Tensor& qb, at::Tensor& partial,
                      at::Tensor& out, const at::Tensor& resid, int64_t fmt_gu, double lim, const at::Tensor& meta);
void gm_q8_swiglu_down(const at::Tensor& g, const at::Tensor& u, const at::Tensor& wd, at::Tensor& out, bool accumulate);
void gm_qkv_decode(const at::Tensor& x, const std::vector<at::Tensor>& ws, const std::vector<int64_t>& fmts, at::Tensor& out, at::Tensor& qb);
void gm_pf_gate_up(const at::Tensor& x, const at::Tensor& gate_ptrs, const at::Tensor& up_ptrs, at::Tensor& H, const at::Tensor& sp, const at::Tensor& be,
                   int64_t P, int64_t topk, int64_t E, double lim, int64_t K, int64_t N, int64_t rowb, int64_t maxb, int64_t fmt);
void gm_pf_down(const at::Tensor& H, const at::Tensor& down_ptrs, const at::Tensor& wts, const at::Tensor& sp, const at::Tensor& be, at::Tensor& out,
                int64_t P, int64_t E, int64_t K, int64_t N, int64_t rowb, int64_t fmt, int64_t maxb);
#endif
