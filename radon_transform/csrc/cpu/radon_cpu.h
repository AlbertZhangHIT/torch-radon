
#pragma once
#include <torch/extension.h>

at::Tensor radon_cpu(const at::Tensor& img, 
					const at::Tensor& theta);