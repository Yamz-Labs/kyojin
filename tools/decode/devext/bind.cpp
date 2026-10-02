#include <torch/extension.h>
#include "quant/exl3_dec.cuh"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemv", &exl3_dec_gemv,
          py::arg("x"), py::arg("trellis"), py::arg("suh"), py::arg("svh"), py::arg("out"),
          py::arg("scratch"), py::arg("counters"), py::arg("K"), py::arg("mcg") = false);
    m.def("gemv_multi", &exl3_dec_gemv_multi,
          py::arg("x"), py::arg("trellis"), py::arg("suh"), py::arg("svh"), py::arg("out"),
          py::arg("scratch"), py::arg("counters"), py::arg("K"), py::arg("mcg") = false);
    m.def("moe", &exl3_dec_moe);
    m.def("gemv_strided", &exl3_dec_gemv_strided,
          py::arg("x"), py::arg("trellis"), py::arg("suh"), py::arg("svh"), py::arg("out"),
          py::arg("scratch"), py::arg("counters"), py::arg("K"), py::arg("in_features"),
          py::arg("x_gstride"), py::arg("mcg") = false);
    m.def("router", &exl3_dec_router);
    m.def("router_norm", &exl3_dec_router_norm);
    m.def("rms_norm", &exl3_dec_rms_norm);
}
