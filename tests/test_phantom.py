import unittest

import torch
import numpy as np
from torchradon import radon, iradon, np_iradon

import matplotlib.pyplot as plt

class TestRADONBIAS(unittest.TestCase):
	def test_radon_bias_circular_phantom(self):
		"""
		test that a uniform circular phantom has a small reconstruction bias
		"""
		pixels = 128
		xy = np.arange(-pixels / 2, pixels / 2) + 0.5
		x, y = np.meshgrid(xy, xy)
		image = x**2 + y**2 <= (pixels/4)**2
		image = np.asarray(image, dtype=np.float)

		theta = np.linspace(0., 180., max(image.shape), endpoint=False)

		image = torch.from_numpy(image).float()
		theta = torch.from_numpy(theta).float()

		sinogram = radon(image, theta=theta)

		fbp = iradon(sinogram, theta=theta, output_size=image.size(0))
		error_fbp = image - fbp
		print("iradon FBP rms error: %.3g, mae error: %.3g" 
			% ((error_fbp**2).mean().sqrt(), error_fbp.abs().mean()))

		np_fbp = np_iradon(sinogram, theta=theta, output_size=image.size(0))
		error_np_fbp = image - np_fbp
		print("np_iradon FBP rms error: %.3g, mae error: %.3g" 
			% ((error_np_fbp**2).mean().sqrt(), error_np_fbp.abs().mean()))


		fig, axarr = plt.subplots(3, 2, figsize=(8, 13.5))
		axarr[0, 0].set_title("Original")
		axarr[0, 0].imshow(image.numpy(), cmap=plt.cm.Greys_r)

		axarr[0, 1].set_title("Radon transform\n(Sinogram)")
		axarr[0, 1].set_xlabel("Projection angle (deg)")
		axarr[0, 1].set_ylabel("Projection position (pixels)")
		axarr[0, 1].imshow(sinogram.numpy(), cmap=plt.cm.Greys_r, 
			extent=(0, 180, 0, sinogram.size(0)), aspect='auto')

		axarr[1, 0].set_title("fbp iradon")
		axarr[1, 0].imshow(fbp.numpy(), cmap=plt.cm.Greys_r)

		axarr[1, 1].set_title("fbp iradon error")
		axarr[1, 1].imshow((error_fbp).numpy(), cmap=plt.cm.Greys_r)		

		axarr[2, 0].set_title("fbp np_iradon")
		axarr[2, 0].imshow(np_fbp.numpy(), cmap=plt.cm.Greys_r)

		axarr[2, 1].set_title("fbp np_iradon error")
		axarr[2, 1].imshow((error_np_fbp).numpy(), cmap=plt.cm.Greys_r)

		fig.tight_layout()
		plt.show()

	def test_radon_bias_phantom(self):
		"""
		test that the phantom image has a small reconstruction bias
		"""		
		from skimage.io import imread
		from skimage.transform import rescale
		image = imread('./phantom.png', as_gray=True) # the pixel value range is [0, 1]
		#image = rescale(image, scale=0.4, mode='reflect', multichannel=False)
		theta = np.linspace(0., 180., max(image.shape), endpoint=False)

		image = torch.from_numpy(image).float()
		theta = torch.from_numpy(theta).float()
		sinogram = radon(image, theta=theta)

		fbp = iradon(sinogram, theta=theta, output_size=image.size(0))
		error_fbp = image - fbp
		print("iradon FBP rms error: %.3g, mae error: %.3g" 
			% ((error_fbp**2).mean().sqrt(), error_fbp.abs().mean()))

		np_fbp = np_iradon(sinogram, theta=theta, output_size=image.size(0))
		error_np_fbp = image - np_fbp
		print("np_iradon FBP rms error: %.3g, mae error: %.3g" 
			% ((error_np_fbp**2).mean().sqrt(), error_np_fbp.abs().mean()))


		fig, axarr = plt.subplots(3, 2, figsize=(8, 13.5))
		axarr[0, 0].set_title("Original")
		axarr[0, 0].imshow(image.numpy(), cmap=plt.cm.Greys_r)

		axarr[0, 1].set_title("Radon transform\n(Sinogram)")
		axarr[0, 1].set_xlabel("Projection angle (deg)")
		axarr[0, 1].set_ylabel("Projection position (pixels)")
		axarr[0, 1].imshow(sinogram.numpy(), cmap=plt.cm.Greys_r, 
			extent=(0, 180, 0, sinogram.size(0)), aspect='auto')

		axarr[1, 0].set_title("fbp iradon")
		axarr[1, 0].imshow(fbp.numpy(), cmap=plt.cm.Greys_r)

		axarr[1, 1].set_title("fbp iradon error")
		axarr[1, 1].imshow((error_fbp).numpy(), cmap=plt.cm.Greys_r)		

		axarr[2, 0].set_title("fbp np_iradon")
		axarr[2, 0].imshow(np_fbp.numpy(), cmap=plt.cm.Greys_r)

		axarr[2, 1].set_title("fbp np_iradon error")
		axarr[2, 1].imshow((error_np_fbp).numpy(), cmap=plt.cm.Greys_r)

		fig.tight_layout()
		plt.show()				

if __name__ == "__main__":
	unittest.main()