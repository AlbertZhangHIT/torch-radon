import os
import torch
from torch.utils.ffi import create_extension

sources = ['src/th_radon.cpp']
headers = ['src/th_radon.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
	pass

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)

ffi = create_extension(
	'_ext.radon',
	headers=headers,
	sources=sources,
	define_macros=defines,
	relative_to=__file__,
	with_cuda=with_cuda,
	extra_compile_args=['-std=c11'])

if __name__ == '__main__':
	ffi.build()