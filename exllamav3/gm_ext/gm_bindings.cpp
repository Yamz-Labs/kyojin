#include <torch/extension.h>
#include <pybind11/stl.h>
#include "gm_router.cuh"
#include "gm_moe.cuh"
#include "gm_dense_q8.cuh"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gm_routing_std_bsz1", &gm_routing_std_bsz1, "gm_routing_std_bsz1");
    m.def("gm_qkv_decode", &gm_qkv_decode, "gm_qkv_decode");
    m.def("gm_moe_kq_decode", &gm_moe_kq_decode, "gm_moe_kq_decode", py::arg("h"), py::arg("sel"), py::arg("wts"), py::arg("gate_ptrs"), py::arg("up_ptrs"),
          py::arg("down_ptrs"), py::arg("gate_stride"), py::arg("up_stride"), py::arg("down_stride"), py::arg("fmt_d"), py::arg("act"), py::arg("qb"),
          py::arg("partial"), py::arg("out"), py::arg("resid"), py::arg("fmt_gu") = 0, py::arg("lim") = 0.0, py::arg("meta") = at::Tensor());
    m.def("gm_q8_swiglu_down", &gm_q8_swiglu_down, "gm_q8_swiglu_down");
    m.def("gm_dense_q8", &gm_dense_q8, "gm_dense_q8");
    m.def("gm_route", &gm_route, "gm_route");
    m.def("gm_pf_gate_up", &gm_pf_gate_up, "gm_pf_gate_up");
    m.def("gm_pf_down", &gm_pf_down, "gm_pf_down");
}
