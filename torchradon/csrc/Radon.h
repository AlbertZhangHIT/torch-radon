#pragma once
#include "cpu/radon_cpu.h"
#include "cpu/backproject_cpu.h"

at::Tensor radon(const at::Tensor& input,
                 const at::Tensor& theta) {
	return radon_cpu(input, theta);
}

at::Tensor backproject(const at::Tensor& radon_img, 
						const at::Tensor& theta,
						const int output_size,
						const int interp_flag) {
	return backproject_cpu(radon_img, 
						theta,
						output_size,
						interp_flag);
}