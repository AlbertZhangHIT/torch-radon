#include "cpu/backproject_cpu.h"

template <typename scalar_t>
at::Tensor backproject_cpu_kernel(const at::Tensor& filtered_proj_t, 
				const at::Tensor& theta_t,
				const int output_size, const int interp_flag) {
	AT_ASSERTM(!filtered_proj_t.type().is_cuda(), "filtered_proj must be a CPU tensor");
	AT_ASSERTM(!theta_t.type().is_cuda(), "theta must be a CPU tensor");	

	if (filtered_proj_t.numel() == 0) {
		return at::empty({0}, filtered_proj_t.options());
	}

	auto len_radius = filtered_proj_t.size(0);
	auto numAngles = theta_t.numel();
	at::Tensor fbp_img_t = at::zeros({output_size, output_size}, filtered_proj_t.options());

	auto filtered_proj = filtered_proj_t.data<scalar_t>();
	auto theta = theta_t.data<scalar_t>();
	auto fbp_img = fbp_img_t.data<scalar_t>();

	auto ctr = std::floor((output_size - 1) / 2);
	auto xleft = -ctr;
	auto ytop = ctr;

	int64_t ctr_idx = (int64_t) std::floor(len_radius / 2);

	for (int64_t k = 0; k < numAngles; k++) {
		auto cos_theta = std::cos(theta[k]);
		auto sin_theta = std::sin(theta[k]);
		//int64_t proj = k * len_radius;
		switch(interp_flag) {
			case 0:
				for (int64_t w = 0; w < output_size; w++) {
					auto t = (xleft + w) * cos_theta + ytop * sin_theta;
					for (int64_t h = 0; h < output_size; h++) {
						int64_t tmp = std::round(t);
						int64_t index = (tmp + ctr_idx)*numAngles + k;
						fbp_img[h*output_size + w] += filtered_proj[index];
						t -= sin_theta;
					} // w
				} // h
				break;

			case 1:
				for (int64_t w = 0; w < output_size; w++) {
					auto t = (xleft + w) * cos_theta + ytop * sin_theta;
					for (int64_t h = 0; h < output_size; h++) {
						int64_t tmp = std::floor(t);
						int64_t index = (tmp + ctr_idx)*numAngles + k;
						int64_t index_1 = (tmp + ctr_idx + 1)*numAngles + k;
						fbp_img[h*output_size + w] += (t - tmp) * (filtered_proj[index_1] - filtered_proj[index]) + filtered_proj[index];
						t -= sin_theta;
					} // w
				} // h
				break;
		} // switch
	} // theta	
	return fbp_img_t;
}

at::Tensor backproject_cpu(const at::Tensor& filtered_proj, 
						const at::Tensor& theta,
						const int output_size,
						const int interp_flag) {
	at::Tensor result;
	AT_DISPATCH_FLOATING_TYPES(filtered_proj.type(), 'backproject', [&] {
		result = backproject_cpu_kernel<scalar_t>(
			filtered_proj, theta,
			output_size, interp_flag);
	});
	return result;
}