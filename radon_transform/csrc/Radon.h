#pragma once
#include "cpu/th_radon.h"
#include "cpu/backproject.h"

at::Tensor radon(const at::Tensor& input,
                 const at::Tensor& theta) {
	return radon_cpu(input, theta);
}

at::Tensor backproject(const at::Tensor& radon_img, 
						const at::Tensor& costheta,
						const at::Tensor& sintheta, 
						const int output_size,
						const int interp_flag) {
	return backproject_cpu(radon_img, 
						costheta,
						sintheta, 
						output_size,
						interp_flag);
}