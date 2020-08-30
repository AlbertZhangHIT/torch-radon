import unittest

import torch
import numpy as np
import scipy.io as sio
from torchradon import radon, iradon

import matplotlib.pyplot as plt

class TestCompare(unittest.TestCase):
	"""Compare the radon transform and iradon reconstruction results with matlab
	"""
	def test_compare_phantom(self):
		"""comparison on phantom image
		"""
		from skimage.io import imread
		from scipy.io import loadmat
		image = imread('./phantom.png', as_gray=True) # the pixel value range is [0, 1]
		theta = np.linspace(0., 180., max(image.shape), endpoint=False)

		image = torch.from_numpy(image).float()
		theta = torch.from_numpy(theta).float()
		sinogram = radon(image, theta=theta)

		matlab_sinogram = loadmat("./phantom-sinogram.mat")['sinogram']
		sinogram_err = sinogram - torch.from_numpy(matlab_sinogram).float()
		
		print("radon sinogram rms error: %.3g, mae error: %.3g" 
			% ((sinogram_err**2).mean().sqrt(), sinogram_err.abs().mean()))

		# reconstruction comparison
		fbp = iradon(sinogram, theta=theta, output_size=None) # without setting output size the shape will match with matlab
		matlab_fbp = loadmat("./phantom-fbp.mat")['fbp']
		fbp_err = fbp - torch.from_numpy(matlab_fbp).float()

		print("radon FBP rms error: %.3g, mae error: %.3g" 
			% ((fbp_err**2).mean().sqrt(), fbp_err.abs().mean()))		

if __name__ == "__main__":
	unittest.main()		