
#include "Radon.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("radon", &radon, "radon transform");
}