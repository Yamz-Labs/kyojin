#include <torch/extension.h>
#include <pybind11/stl.h>
#include "gm_router.cuh"
#include "gm_moe.cuh"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gm_routing_std_bsz1", &gm_routing_std_bsz1, "gm_routing_std_bsz1");
    m.def("gm_qkv_decode", &gm_qkv_decode, "gm_qkv_decode");
    m.def("gm_moe_kq_decode", &gm_moe_kq_decode, "gm_moe_kq_decode");
    m.def("gm_pf_gate_up", &gm_pf_gate_up, "gm_pf_gate_up");
    m.def("gm_pf_down", &gm_pf_down, "gm_pf_down");
}
