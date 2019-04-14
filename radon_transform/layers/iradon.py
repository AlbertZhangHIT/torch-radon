import torch
import numpy as np

from radon_transform import _C

def designFilter(filter_mode, length, d=1.):
	if filter_mode not in ('ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann'):
		raise ValueError('Invalide filter %s seleted.' % filter_mode) 
	order = max(64, int(2 ** np.ceil(np.log2(2*length))))
	filt = 2. * torch.arange(0, order/2+1) / order
	w = 2. * np.pi * torch.arange(0, len(filt)) / order

	if filter_mode == 'ram-lak':
		pass
	elif filter_mode == 'shepp-logan':
		filt[1:len(filt)] = filt[1:len(filt)] * torch.sin(w[1:len(w)] / (2*d)) / (w[1:len(w)] / (2*d))
	elif filter_mode == 'cosine':
		filt[1:len(filt)] = filt[1:len(filt)] * torch.cos(w[1:len(w)] / (2*d))
	elif filter_mode == 'hamming':
		filt[1:len(filt)] = filt[1:len(filt)] * (.54 + .46 * torch.cos(w[1:len(w)] / d))
	elif filter_mode == 'hann':
		filt[1:len(filt)] = filt[1:len(filt)] * (1 + torch.cos(w[1:len(w)] / d)) / 2
	else:
		pass

	filt[w > np.pi*d] = 0
	flip_filt = torch.flip(filt, [0, 1])
	filt = torch.cat([filt, flip_filt[1:len(filt)]])

	return filt

def filterProjections(radon_img, filter_mode, d=1.):
	length = radon_img.size(0)
	H = designFilter(filter_mode, length, d)
	radon_img[len(H)-1, 0] = 0 # zero pad projections
	p = torch.rfft(radon_img, signal_ndim=1, onesided=False)
	H_expand = H.expand([p.size(0), p.size(1)]).unsqueeze(-1).expand(*p.size())
	p = p * H_expand # frequency domain filtering
	p = p[...,0] # real part
	p = p[0:length, :] #Truncate the filtered projection

	return p



def iradon(radon_img, theta=None, output_size=None, filt='ram-lak', interp_mode='linear', d=1.):
	# iradon.__doc__ = """
	# This function performs Filtered Backprojection of 2-D tensor of radon measures"""
	radon_img = radon_img.cpu().squeeze()
	theta = theta.cpu()

	if radon_img.dim() != 2:
		raise ValueError("Only 2-D Tensors are supported.")
	m, n = radon_img.size()
	if theta is None:
		theta = torch.linspace(0., 180.*(1.-1./n), n)
	if interp_mode not in ('linear', 'nearest'):
		raise ValueError("Unknown interpolation: %s" % interp_mode)
	if interp_mode == 'linear':
		interp_flag = 1
	elif interp_mode == 'nearest':
		interp_flag = 0
	else:
		pass
	if not output_size:
		output_size = int(2 * np.floor( m / np.sqrt(2.) / 2.))

	filtered_proj = filterProjections(radon_img, filt, d)

	img = torch.zeros(output_size, output_size)

	costheta = torch.cos(theta)
	sintheta = torch.sine(theta)
	img = _C.backproject(filtered_proj, costheta, sintheta, interp_flag)

	return img


