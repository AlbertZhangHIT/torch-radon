#include "cpu/backproject.h"

template <typename T>
void backproject_cpu_kernel(T *img, const T *radon_img, 
							const T *theta,
							const int measure_len, const int num_proj,
							const int output_size, const int interp_flag) {

	T cos_theta, sin_theta, t;
	const T *proj;
	T * imgPtr;
	int tmp;

	T ctr = std::floor((output_size-1) / 2);
	T xleft = -ctr;
	T ytop = ctr;

	int ctr_idx = (int) std::floor(measure_len / 2);
	int ctr_idx_1 = ctr_idx + 1;

	for (int theta_idx = 0; theta_idx < num_proj; theta_idx++)
	{
		cos_theta = std::cos(theta[theta_idx]);
		sin_theta = std::sin(theta[theta_idx]);
		proj = (radon_img + theta_idx*measure_len);
		imgPtr = img;
		switch (interp_flag)
		{
		case 0: /* nearest-neighbour interpolation */
			for (int x_idx = 0; x_idx < output_size; x_idx++)
			{
				t = (xleft + x_idx) * cos_theta + ytop * sin_theta;
				for (int y_idx = 0; y_idx < output_size; y_idx++)
				{
					tmp = std::round(t);
					*imgPtr++ += proj[tmp + ctr_idx];
					t -= sin_theta;
				} /* end of y-loop*/
			} /* end of x-loop*/
			break;

		case 1: /* linear interpolation */
			for (int x_idx = 0; x_idx < output_size; x_idx++)
			{
				t = (xleft + x_idx) * cos_theta + ytop * sin_theta;
				for (int y_idx = 0; y_idx < output_size; y_idx++)
				{
					tmp = std::floor(t);
					*imgPtr++ += (t-tmp) * (proj[tmp + ctr_idx_1] - proj[tmp + ctr_idx]) + proj[tmp + ctr_idx];
					t -= sin_theta;
				} /* end of y-loop*/
			} /* end of x-loop*/
			break;			
		}
	}

}

at::Tensor backproject_cpu(const at::Tensor& radon_img, 
						const at::Tensor& theta,
						const int output_size,
						const int interp_flag) {
	AT_ASSERTM(!radon_img.type().is_cuda(), "radon measure must be a CPU tensor");
	AT_ASSERTM(!theta.type().is_cuda(), "theta must be a CPU tensor");


	auto measure_len = radon_img.size(0);
	auto num_proj = radon_img.size(1);

	auto fbp_img = at::zeros({output_size, output_size}, radon_img.options());

	AT_DISPATCH_FLOATING_TYPES(radon_img.type(), 'backproject', [&] {
		backproject_cpu_kernel<scalar_t>(
			fbp_img.data<scalar_t>(),
			radon_img.data<scalar_t>(), 
			theta.data<scalar_t>(),
			measure_len, num_proj,
			output_size, interp_flag);
	});
	return fbp_img;
}