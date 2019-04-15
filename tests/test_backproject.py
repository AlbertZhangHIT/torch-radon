import unittest

import torch
import numpy as np
import scipy.io as sio
from radon_transform._C import backproject


class TestBackproject(unittest.TestCase):
	def test_backproject_cpu(self):
		filtered_proj = sio.loadmat('p.mat')['p']
		theta = np.linspace(0, 180, 50, endpoint=False)
		target = sio.loadmat('sparse.mat')['sparse']

		theta = torch.from_numpy(theta).float() * np.pi / 180.
		filtered_proj = torch.from_numpy(filtered_proj).float()
		target = torch.from_numpy(target).float()

		print("filtered_proj(0, 0) = %f\n" % filtered_proj[0, 0])
		print("filtered_proj(0, 1) = %f\n" % filtered_proj[0, 1])
		print("filtered_proj(0, 2) = %f\n" % filtered_proj[0, 2])
		print("filtered_proj(0, 3) = %f\n" % filtered_proj[0, 3])

		fbp = backproject(filtered_proj, theta, 512, 1)
		fbp = fbp * np.pi / (2*theta.numel())
		sio.savemat('torch_fbp.mat', dict(fbp=fbp.numpy()))

		error = target - fbp
		tol = 5e-5
		roi_err = error.mean().abs()
		norm_err = torch.norm(error, 2)
		print("Error: %.3f, Norm: %.3f" % (roi_err, norm_err))
		assert(roi_err < tol)		

if __name__ == "__main__":
	unittest.main()