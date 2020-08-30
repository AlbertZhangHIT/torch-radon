import torch
import numpy as np
from scipy.interpolate import interp1d
from torchradon import _C

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
	flip_filt = torch.flip(filt, [-1])
	filt = torch.cat([filt, flip_filt[1:len(filt)-1]])

	return filt

def filterProjections(radon_img, filter_mode, d=1.):
	length = radon_img.size(0)
	H = designFilter(filter_mode, length, d)
	p = torch.zeros(len(H), radon_img.size(1), 2) # p holds fft of projections
	p[0:length, :, 0] = radon_img # zero pad

	fp = torch.fft(p.permute(1,0,2), signal_ndim=1)

	H_expand = H.unsqueeze(0).expand([fp.size(0), fp.size(1)]).unsqueeze(-1).expand(*fp.size())
	fp = fp * H_expand # frequency domain filtering
	p = torch.ifft(fp, signal_ndim=1).permute(1,0,2)
	p = p[...,0] # real part
	p = p[0:length, :] #Truncate the filtered projection

	return p.contiguous() # method 'contiguous' is vitally important, if not it will cause memory leaking



def iradon(radon_img, theta=None, output_size=None, filt='ram-lak', interp_mode='linear', d=1.):
	# iradon.__doc__ = """
	# This function performs Filtered Backprojection of 2-D tensor of radon measures
	# using the torch implementation
	# """
	radon_img = radon_img.cpu().squeeze()

	if radon_img.dim() != 2:
		raise ValueError("Only 2-D Tensors are supported.")
	m, n = radon_img.size()
	if theta is None:
		theta = torch.linspace(0., 180.*(1.-1./n), n) * np.pi / 180.
	theta = theta.cpu() * np.pi / 180.
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

	img = _C.backproject(filtered_proj, theta, output_size, interp_flag)

	img = img * np.pi / (2.*theta.numel())

	return img

def np_iradon(radon_img, theta=None, output_size=None, filt='ram-lak', interp_mode='linear', d=1.):
	# iradon.__doc__ = """
	# This function performs Filtered Backprojection of 2-D tensor of radon measures
	# using the numpy pipeline implementation
	# """
	radon_img = radon_img.cpu().squeeze()

	if radon_img.dim() != 2:
		raise ValueError("Only 2-D Tensors are supported.")
	m, n = radon_img.size()
	if theta is None:
		theta = torch.linspace(0., 180.*(1.-1./n), n) * np.pi / 180.
	theta = theta.cpu() * np.pi / 180.
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

	reconstructed = np.zeros((output_size, output_size))
	# Determine the center of the projections (= center of sinogram)
	mid_index = m // 2

	[Y, X] = np.mgrid[0:output_size, 0:output_size]
	#[Y, X] = np.mgrid[1:output_size+1, 1:output_size+1]
	xpr = X - int(output_size) // 2
	ypr = Y - int(output_size) // 2	
	# Reconstruct image by interpolation
	for i in range(len(theta)):
		t = -ypr * np.sin(theta[i]).numpy() + xpr * np.cos(theta[i]).numpy()
		taxis = np.arange(filtered_proj.size(0)) - mid_index
		#taxis = np.arange(1, filtered_proj.size(0)+1) - mid_index
		if interp_mode == 'linear':
			backprojected = np.interp(t, taxis, filtered_proj[:, i].numpy(),
								left=0, right=0)
		else:
			interpolant = interp1d(taxis, filtered_proj[:, i].numpy(), kind=interp_mode,
						bounds_error=False, fill_value=0)
			backprojected = interpolant(t)
		reconstructed += backprojected

	return torch.from_numpy(reconstructed).float() * np.pi / (2 * len(theta))	


