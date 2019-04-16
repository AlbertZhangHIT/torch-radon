import unittest

import torch
import numpy as np
import scipy.io as sio
from radon_transform.layers import radon, iradon

import matplotlib.pyplot as plt

class TestIRADON(unittest.TestCase):
	def test_iradon_cpu(self):
		radon_img = sio.loadmat('rdQ.mat')['rdQ']
#		theta = sio.loadmat('theta.mat')['theta']
		theta = np.linspace(0., 180., 50, endpoint=False)
		target = sio.loadmat('sparse.mat')['sparse']

		radon_img = torch.from_numpy(radon_img).float()
		theta = torch.from_numpy(theta).float()
		target = torch.from_numpy(target).float()

		recon_fbp = iradon(radon_img, theta, output_size=512)
		error = recon_fbp - target

		sio.savemat('torch_iradon.mat', dict(fbp_img=recon_fbp.numpy()))

		fig, axarr = plt.subplots(2, 2, figsize=(8, 13.5))
		axarr[0, 0].set_title("Target")
		axarr[0, 0].imshow(target.numpy(), cmap=plt.cm.Greys_r)

		axarr[0, 1].set_title("Radon transform\n(Sinogram)")
		axarr[0, 1].set_xlabel("Projection angle (deg)")
		axarr[0, 1].set_ylabel("Projection position (pixels)")
		axarr[0, 1].imshow(radon_img.numpy(), cmap=plt.cm.Greys_r, 
			extent=(0, 180, 0, radon_img.size(0)), aspect='auto')

		axarr[1, 0].set_title("FBP reconstruction")
		axarr[1, 0].imshow(recon_fbp.numpy(), cmap=plt.cm.Greys_r)		
		axarr[1, 1].set_title("FBP reconstruction error")
		axarr[1, 1].imshow((error).numpy(), cmap=plt.cm.Greys_r)
		fig.tight_layout()
		plt.show()
		
		tol = 5e-5
		roi_err = error.mean().abs()
		norm_err = torch.norm(error, 2)
		print("Error: %.3f, Norm: %.3f" % (roi_err, norm_err))
		#assert(roi_err < tol)

if __name__ == "__main__":
	unittest.main()