#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <cmath>

namespace py = pybind11;

// Multiply psi by exp(-i * coeff * (V + g*|psi|^2)) in-place.
void apply_nonlinear_phase(
    py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> psi,
    py::array_t<float, py::array::c_style | py::array::forcecast> V,
    float g,
    float coeff
) {
    auto p = psi.mutable_unchecked<2>();
    auto v = V.unchecked<2>();

    const ssize_t nx = p.shape(0);
    const ssize_t ny = p.shape(1);

    for (ssize_t i = 0; i < nx; ++i) {
        for (ssize_t j = 0; j < ny; ++j) {
            std::complex<float> z = p(i, j);
            float amp2 = std::norm(z);
            float theta = -coeff * (v(i, j) + g * amp2);
            std::complex<float> ph(std::cos(theta), std::sin(theta));
            p(i, j) = ph * z;
        }
    }
}

PYBIND11_MODULE(scalar_bec_cpp, m) {
    m.doc() = "C++ kernels for scalar-BEC";
    m.def("apply_nonlinear_phase", &apply_nonlinear_phase,
          "Apply nonlinear phase in-place",
          py::arg("psi"), py::arg("V"), py::arg("g"), py::arg("coeff"));
}
