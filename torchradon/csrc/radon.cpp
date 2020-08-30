
#include "Radon.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("radon", &radon, "radon transform");
	m.def("backproject", &backproject, "backprojection for fbp");
}