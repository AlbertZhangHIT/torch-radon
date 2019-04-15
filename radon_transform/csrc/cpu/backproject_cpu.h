#pragma once
#include <torch/extension.h>

at::Tensor backproject_cpu(const at::Tensor& radon_img, 
						const at::Tensor& theta,
						const int output_size,
						const int interp_flag);