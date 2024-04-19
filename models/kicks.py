import torch
from torch import nn
from models.utils.layers import ConvLayer, DeconvLayer


class Generator(nn.Module):
	def __init__(self, nz: int = 20,
				 ngf: int = 48,
				 nc: int = 1,
				 normalization: tuple = ('batch_norm', {}),
				 activation: str = 'relu'):

		super(Generator, self).__init__()
		self.constructor = {
			'nz': nz,
			'ngf': ngf,
			'nc': nc,
			'normalization': normalization,
			'activation': activation
		}

		# pre-processing arguments
		if normalization is None:
			normalization = None, None

		bias = normalization[0] not in {'spectral_norm', None}

		if normalization[0] not in {'spectral_norm', None}:
			last_norm = None
		else:
			last_norm = normalization

		# parameters
		self.nz = nz
		self.ngf = ngf
		self.nc = nc
		h_size = 1, 32, 128, 512, 2048, 8192, 32768
		h_chan = nz, 16*ngf, 8*ngf, 4*ngf, 2*ngf, ngf, nc
		h_kern = 32, 8, 8, 8, 8, 8
		h_strd = 1, 4, 4, 4, 4, 4
		h_padd = 0, 2, 2, 2, 2, 2
		h_bias = [bias] * (len(h_kern)-1) + [True]
		h_norm = [normalization] * (len(h_kern)-1) + [last_norm]
		h_actv = [activation] * (len(h_kern)-1) + ['tanh']

		for k in range(6):
			setattr(self, f'layer_{k}', DeconvLayer(in_channels=h_chan[k],
													out_channels=h_chan[k + 1],
													kernel_size=h_kern[k],
													stride=h_strd[k],
													padding=h_padd[k],
													bias=h_bias[k],
													normalization=h_norm[k],
													activation=h_actv[k],
													out_length=h_size[k]))

	def forward(self, x: torch.Tensor=None):
		if x is None:
			x = self.make_inputs(batchsize=1)
		for k in range(6):
			x = getattr(self, f'layer_{k}')(x)
		return x

	def make_inputs(self, batchsize: int, device: str = 'cpu'):
		return torch.randn(batchsize, self.nz, 1).to(device)


class Encoder(nn.Module):
	def __init__(self, nc: int = 1,
				 ngf: int = 48,
				 nz: int = 20,
				 normalization: tuple = ('batch_norm', {}),
				 activation: str = 'relu'):

		super(Encoder, self).__init__()
		self.constructor = {
			'nc': nc,
			'ngf': ngf,
			'nz': nz,
			'normalization': normalization,
			'activation': activation
		}

		# pre-processing arguments
		if normalization is None:
			normalization = None, None

		bias = normalization[0] not in {'spectral_norm', None}

		if normalization[0] not in {'spectral_norm', None}:
			last_norm = None
		else:
			last_norm = normalization

		# parameters
		self.nz = nz
		self.ngf = ngf
		self.nc = nc
		h_size = 32768, 8192, 2048, 512, 128, 32, 1
		h_chan = nc, ngf, 2*ngf, 4*ngf, 8*ngf, 16*ngf, nz
		h_kern = 8, 8, 8, 8, 8, 32
		h_strd = 4, 4, 4, 4, 4, 1
		h_padd = 2, 2, 2, 2, 2, 0
		h_norm = [normalization] * (len(h_kern)-1) + [last_norm]
		h_actv = [activation] * (len(h_kern)-1) + [None]
		h_bias = [bias] * (len(h_kern)-1) + [True]
		for k in range(6):
			setattr(self, f'layer_{k}', ConvLayer(in_channels=h_chan[k],
												  out_channels=h_chan[k + 1],
												  kernel_size=h_kern[k],
												  stride=h_strd[k],
												  padding=h_padd[k],
												  bias=h_bias[k],
												  normalization=h_norm[k],
												  activation=h_actv[k],
												  out_shape=h_size[k + 1]))

	def forward(self, x: torch.Tensor):
		for k in range(6):
			x = getattr(self, f'layer_{k}')(x)
		return x


if __name__ == '__main__':
	pass
