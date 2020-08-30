from torchradon import _C
import numpy as np

def radon(img, theta=None):
	# radon.__doc__ = """
	# This function performs radon transform of 2-D tensor 
	# using the torch implementation
	# """
	img = img.cpu().squeeze()

	if img.dim() != 2:
		raise ValueError("Only 2-D Tensors are supported.")
	m, n = img.size()
	if theta is None:
		theta = torch.linspace(0., 180.*(1.-1./n), n)
	theta = theta.cpu()	* np.pi / 180.
	radon_img = _C.radon(img, theta)
	return radon_img