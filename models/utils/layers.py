import torch
from torch import nn
from models.utils.activations import get_activation, Identity


class ConvLayer(nn.Module):
	def __init__(self,
				 in_channels: int,
				 out_channels: int,
				 kernel_size: int = 3,
				 stride: int = 1,
				 padding: int = 0,
				 bias: bool = True,
				 normalization: tuple = (None, None),
				 activation: str = 'lrelu',
				 out_shape: int = None):

		super(ConvLayer, self).__init__()

		# pre-process normalization:
		if normalization is None:
			normalization = None, None
		norm_mode, norm_param = normalization

		if norm_mode in {'batch_norm',
						 'instance_norm',
						 'layer_norm',
						 'local_response_norm'}:
			bias = False
		elif norm_mode in {'spectral_norm', None}:
			bias = True and bias
		else:
			raise ValueError(f"First element of normalization ({norm_mode}) "
							 f"must be 'batch_norm', 'layer_norm', "
							 f"'instance_norm', 'local_response_norm' or "
							 f"'spectral_norm'.")

		# convolution
		self.conv = nn.Conv1d(in_channels=in_channels,
							  out_channels=out_channels,
							  kernel_size=kernel_size,
							  stride=stride,
							  padding=padding,
							  bias=bias)

		# normalization
		if norm_mode == 'batch_norm':
			self.norm = nn.BatchNorm1d(num_features=out_channels, **norm_param)
		elif norm_mode == 'instance_norm':
			self.norm = nn.InstanceNorm1d(num_features=out_channels, **norm_param)
		elif norm_mode == 'layer_norm':
			self.norm = nn.LayerNorm(normalized_shape=out_shape, **norm_param)
		elif norm_mode == 'local_response_norm':
			self.norm = nn.LocalResponseNorm(**norm_param)
		elif norm_mode == 'spectral_norm':
			self.conv = nn.utils.spectral_norm(self.conv, **norm_param)
			self.norm = Identity()
		elif norm_mode is None:
			self.norm = Identity()

		# activation
		self.activation = get_activation(activation)

	def forward(self, x: torch.Tensor):
		a = self.conv(x)
		a = self.norm(a)
		return self.activation(a)


class DeconvLayer(nn.Module):
	def __init__(self,
				 in_channels: int,
				 out_channels: int,
				 kernel_size: int = 3,
				 stride: int = 1,
				 padding: int = 0,
				 bias: bool = True,
				 normalization: tuple = (None, None),
				 activation: str = 'relu',
				 out_length: int = None):

		super(DeconvLayer, self).__init__()

		# pre-process normalization:
		if normalization is None:
			normalization = None, None
		norm_mode, norm_param = normalization

		if norm_mode in {'batch_norm', 'layer_norm', 'instance_norm',
						 'local_response_norm'}:
			bias = False
		elif norm_mode in {'spectral_norm', None}:
			bias = True and bias
		else:
			raise ValueError(f"First element of normalization ({norm_mode}) "
							 f"must be 'batch_norm', 'layer_norm', "
							 f"'instance_norm', 'local_response_norm' or "
							 f"'spectral_norm'.")

		# convolution
		self.conv = nn.ConvTranspose1d(in_channels=in_channels,
									   out_channels=out_channels,
									   kernel_size=kernel_size,
									   stride=stride,
									   padding=padding,
									   bias=bias)

		# normalization
		if norm_mode == 'batch_norm':
			self.norm = nn.BatchNorm1d(num_features=out_channels, **norm_param)
		elif norm_mode == 'instance_norm':
			self.norm = nn.InstanceNorm1d(num_features=out_channels, **norm_param)
		elif norm_mode == 'layer_norm':
			self.norm = nn.LayerNorm(normalized_shape=out_length, **norm_param)
		elif norm_mode == 'local_response_norm':
			self.norm = nn.LocalResponseNorm(**norm_param)
		elif norm_mode == 'spectral_norm':
			self.conv = nn.utils.spectral_norm(self.conv, **norm_param)
			self.norm = Identity()
		elif norm_mode is None:
			self.norm = Identity()

		# activation
		self.activation = get_activation(activation)

	def forward(self, x: torch.Tensor):
		a = self.conv(x)
		a = self.norm(a)
		return self.activation(a)

