
#include "cpu/radon_cpu.h"

#define MAXX(x,y) ((x) > (y) ? (x) : (y))  

template <typename scalar_t>
at::Tensor radon_cpu_kernel(const at::Tensor& img, 
					const at::Tensor& theta) {
	AT_ASSERTM(!img.type().is_cuda(), "img must be a CPU tensor");
	AT_ASSERTM(!theta.type().is_cuda(), "theta must be a CPU tensor");

	if (img.numel() == 0) {
		return at::empty({0}, img.options());
	}
	auto H = img.size(0);
	auto W = img.size(1);
	auto xOrigin = MAXX(0, (W-1)/2);
	auto yOrigin = MAXX(0, (H-1)/2);
	auto temp1 = H - 1 - yOrigin;
	auto temp2 = W - 1 - xOrigin;
	int64_t rLast = std::ceil(std::sqrt((float) (temp1*temp1+temp2*temp2))) + 1;
	int64_t rFirst = -rLast;
	int64_t rSize = rLast - rFirst + 1;  
	int64_t numAngles = theta.numel();

	at::Tensor xCosTable_t = at::zeros({2*W}, img.options());
	at::Tensor ySineTable_t = at::zeros({2*H}, img.options());
	at::Tensor radon_img_t = at::zeros({rSize, numAngles}, img.options());
	//at::Tensor radon_img_t = at::zeros({numAngles, rSize}, img.options());

	auto xCosTable = xCosTable_t.data<scalar_t>();
	auto ySinTable = ySineTable_t.data<scalar_t>();
	auto radon_img = radon_img_t.data<scalar_t>();
	auto imgPtr = img.data<scalar_t>();

	auto thetaPtr = theta.data<scalar_t>();
	for (int64_t k = 0; k < numAngles; k++) {
		auto angle = thetaPtr[k];
		auto cosine = std::cos(angle);
		auto sine = std::sin(angle);
		for (int64_t w = 0; w < W; w++) {
			auto x = w - xOrigin;
			xCosTable[2*w]   = (x - 0.25)*cosine;  
			xCosTable[2*w+1] = (x + 0.25)*cosine;  
		}
		for (int64_t h = 0; h < H; h++) {
			auto y = yOrigin - h;
			ySinTable[2*h] = (y - 0.25)*sine;
			ySinTable[2*h+1] = (y + 0.25)*sine;
		}

		for (int64_t h = 0; h < H; h++) {
			for (int64_t w = 0; w < W; w++) {
				auto pixel = imgPtr[w*H+h];
				if (pixel != 0.0) {
					pixel *= 0.25;

					auto r1 = xCosTable[2*w] + ySinTable[2*h] - rFirst;
					int64_t rr1 = (int64_t) r1; 
					auto delta1 = r1 - rr1;
					radon_img[k + rr1*numAngles] += pixel * (1. - delta1);
					radon_img[k + (rr1 + 1)*numAngles] += pixel * delta1;
					//radon_img[k*rSize + rr1] += pixel * (1. - delta1);
					//radon_img[k*rSize + (rr1 + 1)] += pixel * delta1;

					auto r2 = xCosTable[2*w+1] + ySinTable[2*h] - rFirst;
					int64_t rr2 = (int64_t) r2; 
					auto delta2 = r2 - rr2;
					radon_img[k + rr2*numAngles] += pixel * (1. - delta2);
					radon_img[k + (rr2 + 1)*numAngles] += pixel * delta2;
					//radon_img[k*rSize + rr2] += pixel * (1. - delta2);
					//radon_img[k*rSize + (rr2 + 1)] += pixel * delta2;

					auto r3 = xCosTable[2*w] + ySinTable[2*h+1] - rFirst;
					int64_t rr3 = (int64_t) r3; 
					auto delta3 = r3 - rr3;
					radon_img[k + rr3*numAngles] += pixel * (1. - delta3);
					radon_img[k + (rr3 + 1)*numAngles] += pixel * delta3;
					//radon_img[k*rSize + rr3] += pixel * (1. - delta3);
					//radon_img[k*rSize + (rr3 + 1)] += pixel * delta3;

					auto r4 = xCosTable[2*w+1] + ySinTable[2*h+1] - rFirst;
					int64_t rr4 = (int64_t) r4; 
					auto delta4 = r4 - rr4;	
					radon_img[k + rr4*numAngles] += pixel * (1. - delta4);
					radon_img[k + (rr4 + 1)*numAngles] += pixel * delta4;
					//radon_img[k*rSize + rr4] += pixel * (1. - delta4);
					//radon_img[k*rSize + (rr4 + 1)] += pixel * delta4;																			
				} // if
			} // for w
		} // for h
	} // for k
	return radon_img_t;
}

at::Tensor radon_cpu(const at::Tensor& img, 
					const at::Tensor& theta) {
	at::Tensor result;
	AT_DISPATCH_FLOATING_TYPES(img.type(), "radon", [&] {
		result = radon_cpu_kernel<scalar_t>(img, theta);
	});
	return result;	
}