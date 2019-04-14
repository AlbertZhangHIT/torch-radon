#pragma once
#include "cpu/th_radon.h"

at::Tensor radon(const at::Tensor& input,
                 const at::Tensor& theta) {
	return radon_cpu(input, theta);
}