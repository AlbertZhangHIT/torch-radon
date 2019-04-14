import unittest

import torch
import numpy as np
import scipy.io as sio
from radon_transform import radon, iradon

class TestIRADON(unittest.TestCase):
	def test_iradon_cpu(self):
		radon_img = sio.loadmat('rdQ.mat')['rdQ']
		theta = np.linspace(0., 180., 50, endpoint=False) * np.pi / 180.
		target = sio.loadmat('sparse.mat')['sparse']

		radon_img = torch.from_numpy(radon_img)
		theta = torch.from_numpy(theta)

		recon_fbp = iradon(radon_img, theta, output_size=512)
		error = recon_fbp - target

		tol = 5e-5
		roi_err = error.mean().abs()
		print("Error: %.3f" % roi_err)
		assert(roi_err < tol)

if __name__ == "__main__":
	unittest.main()