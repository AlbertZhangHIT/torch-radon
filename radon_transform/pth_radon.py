import torch
from ._ext import th_radon
import numpy as np
import math

def radon(img, theta):
	img = img.cpu().squeeze()
	theta = theta.cpu()

	if img.dims() > 2:
		raise ValueError('Only 2D Tensors are supported.')

	M, N = img.size()
	xOrigin = math.floor(max(0, (N-1)/2))
	yOrigin = math.floor(max(0, (N-1)/2))
	temp1 = M - 1 - yOrigin   
	temp2 = N - 1 - xOrigin
	rLast = math.ceil(np.sqrt(temp1*temp1+temp2*temp2)) + 1
	rSize = 2*rLast + 1

	radon_img = torch.FloatTensor(rSize, theta.numel())
	radius = torch.zeros_like(theta)

	th_radon.cpu_radon(radon_img, radius, img, theta)

	return radon_img, radius