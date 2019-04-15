
import unittest

import torch
import numpy as np
import scipy.io as sio
from radon_transform.layers import radon

class TestRADON(unittest.TestCase):
	def test_radon_cpu(self):
		inputs = sio.loadmat('dQ.mat')['dQ']
		theta = np.linspace(0., 180., 50, endpoint=False)
		target = sio.loadmat('rdQ.mat')['rdQ']

		img = torch.from_numpy(inputs).float()
		theta = torch.from_numpy(theta).float()

		radon_img = radon(img, theta)
		sio.savemat('torch_radon.mat', dict(radon_img=radon_img.numpy()))

		error = radon_img - torch.from_numpy(target).float()
		tol = 5e-5
		roi_err = error.mean().abs()
		norm_err = torch.norm(error, 2)
		print("Error: %.3f, Norm: %.3f" % (roi_err, norm_err))
		assert(roi_err < tol)		
		#np.testing.assert_array_equal(measure.numpy(), target)

if __name__ == "__main__":
	unittest.main()