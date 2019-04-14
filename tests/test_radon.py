
import unittest

import torch
import numpy as np
import scipy.io as sio
from radon_transform.layers import radon

class TestRADON(unittest.TestCase):
	def test_radon_cpu(self):
		inputs = sio.loadmat('dQ.mat')['dQ']
		theta = np.linspace(0., 180., 50, endpoint=False) * np.pi / 180.
		target = sio.loadmat('rdQ.mat')['rdQ']

		img = torch.from_numpy(inputs)
		theta = torch.from_numpy(theta)

		measure = radon(img, theta)
		np.testing.assert_array_equal(measure.numpy(), target)

if __name__ == "__main__":
	unittest.main()