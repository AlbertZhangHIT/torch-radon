import unittest

import torch
import numpy as np
import scipy.io as sio
from radon_transform.layers import np_iradon

class TestNPIRADON(unittest.TestCase):
	def test_npiradon_cpu(self):
		radon_img = sio.loadmat('rdQ.mat')['rdQ']
#		theta = sio.loadmat('theta.mat')['theta']
		theta = np.linspace(0., 180., 50, endpoint=False)
		target = sio.loadmat('sparse.mat')['sparse']

		radon_img = torch.from_numpy(radon_img).float()
		theta = torch.from_numpy(theta).float()

		recon_fbp = np_iradon(radon_img, theta, output_size=512)
		sio.savemat('torch_np_iradon.mat', dict(fbp_img=recon_fbp.numpy()))

		error = recon_fbp - torch.from_numpy(target).float()

		tol = 5e-5
		roi_err = error.mean().abs()
		norm_err = torch.norm(error, 2)
		print("Error: %.3f, Norm: %.3f" % (roi_err, norm_err))
		assert(roi_err < tol)

if __name__ == "__main__":
	unittest.main()