import os
import math
import torch
from torch import nn
import torchaudio
import torchaudio.transforms as tf
from models.utils.io import load_model


class ImpactDrums:
	def __init__(self,
				 net_g: nn.Module = 'pretrained/kick_g',
				 net_z: nn.Module = 'pretrained/kick_z',
				 center: torch.Tensor = None,
				 x: torch.Tensor = None,
				 y: torch.Tensor = None,
				 angle_x: float = 0.0,
				 angle_y: float = 0.0,
				 angle_max: float = math.pi/2,
				 loudness: float = 0.0,
				 temp_dir: str = 'tmp',
				 wav_path: str = None,
				 history_path: str = None):

		if isinstance(net_g, str):
			net_g = load_model(net_g)
			net_g.eval()
		if isinstance(net_z, str):
			net_z = load_model(net_z)
			net_z.eval()

		os.makedirs(temp_dir, exist_ok=True)
		if wav_path is None:
			wav_path = os.path.abspath(os.path.join(temp_dir, 'signal.wav'))
		if history_path is None:
			history_path = os.path.abspath(os.path.join(temp_dir, 'history'))

		self.net_g = net_g
		self.net_z = net_z
		self.center = center
		self.x = x
		self.y = y
		self.angle_x = angle_x
		self.angle_y = angle_y
		self.angle_max = angle_max
		self.loudness = loudness
		self.wav_path = wav_path
		self.history_path = history_path
		self.max_history_len = 500
		self._min_norm = 1e-4
		self.preprocess = tf.Compose([
			tf.DownmixMono(channels_first=True),
			tf.PadTrim(1 << 15),
			fading
		])

		if os.path.exists(history_path):
			history = torch.load(history_path)
		else:
			history = []
		self.history = history

		self.generate_plane()
		signal = self.generate_signal()
		torchaudio.save(self.wav_path, signal, 44100)

	def generate_plane(self,
					   center: torch.Tensor = None,
					   x: torch.Tensor = None,
					   y: torch.Tensor = None):

		assert self.net_g is not None, "No generator defined"

		if center is None:
			center = self.net_g.make_inputs(batchsize=1, device='cpu')
		if x is None:
			x = self.net_g.make_inputs(batchsize=1, device='cpu')
		if y is None:
			y = self.net_g.make_inputs(batchsize=1, device='cpu')

		center = center / center.norm()
		x = x - center * torch.dot(x.view(-1), center.view(-1))
		y = y - center * torch.dot(y.view(-1), center.view(-1))
		y = y - x * torch.dot(x.view(-1), y.view(-1))
		x = x / x.norm()
		y = y / y.norm()
		angle_x = 0.0
		angle_y = 0.0

		self.center = center
		self.x = x
		self.y = y
		self.angle_x = angle_x
		self.angle_y = angle_y

	def recenter(self):
		new_center = self.generate_latent_vector()
		new_center = new_center / new_center.norm()
		a = self.center
		b = new_center - a * torch.dot(a.view(-1), new_center.view(-1))
		b = b / b.norm()
		cos_theta = torch.dot(self.center.view(-1), new_center.view(-1))
		sin_theta = torch.sin(torch.acos(cos_theta))

		# change x axis
		x_a = torch.dot(self.x.view(-1), a.view(-1))
		x_b = torch.dot(self.x.view(-1), b.view(-1))
		x_o = self.x - x_a * a - x_b * b
		new_x = x_o + x_a * (a * cos_theta + b * sin_theta) + x_b * (b * cos_theta - a * sin_theta)
		angle_x = 0.0

		# change y axis
		y_a = torch.dot(self.y.view(-1), a.view(-1))
		y_b = torch.dot(self.y.view(-1), b.view(-1))
		y_o = self.y - y_a * a - y_b * b
		new_y = y_o + y_a * (a * cos_theta + b * sin_theta) + y_b * (b * cos_theta - a * sin_theta)
		angle_y = 0.0

		self.center = new_center
		self.x = new_x
		self.y = new_y
		self.angle_x = angle_x
		self.angle_y = angle_y

	def change_center(self):
		new_center = self.net_g.make_inputs(batchsize=1)
		new_center = new_center - self.x * torch.dot(new_center.view(-1), self.x.view(-1))
		new_center = new_center - self.y * torch.dot(new_center.view(-1), self.y.view(-1))
		new_center = new_center / new_center.norm()
		self.center = new_center

	def change_x_axis(self):
		new_x = self.net_g.make_inputs(batchsize=1)
		new_x = new_x - self.center * torch.dot(new_x.view(-1), self.center.view(-1))
		new_x = new_x - self.y * torch.dot(new_x.view(-1), self.y.view(-1))
		new_x = new_x / new_x.norm()
		self.x = new_x

	def change_y_axis(self):
		new_y = self.net_g.make_inputs(batchsize=1)
		new_y = new_y - self.center * torch.dot(new_y.view(-1), self.center.view(-1))
		new_y = new_y - self.x * torch.dot(new_y.view(-1), self.x.view(-1))
		new_y = new_y / new_y.norm()
		self.y = new_y

	def change_angles(self, angle_x: float, angle_y: float):
		assert abs(angle_x) <= self.angle_max, f"angle_x is too big ({angle_x})"
		assert abs(angle_y) <= self.angle_max, f"angle_y is too big ({angle_y})"
		self.angle_x = angle_x
		self.angle_y = angle_y

	def change_angle_max(self, angle_max):
		self.angle_max = angle_max

	def change_loudness(self, loudness: float):
		self.loudness = loudness

	def generate_signal(self):
		assert self.center is not None, "Center of plane is not define"
		assert self.x is not None, "x axis of plane is not define"
		assert self.y is not None, "y axis of plane is not define"
		assert self.angle_max is not None, "Size of plane is not define"

		z = self.generate_latent_vector()
		with torch.no_grad():
			signal = self.net_g(z).squeeze(0)
		signal = fading(signal)
		self.update_history()
		return signal

	def generate_latent_vector(self):
		z = spherical_interpolation(self.center, self.x, self.angle_x)
		z = spherical_interpolation(z, self.y, self.angle_y)
		z = z * self.convert_loudness_to_norm(self.loudness)
		return z

	def encode_signal(self, signal: torch.Tensor):
		signal = self.preprocess(signal)
		assert self.net_z is not None, "No encoder defined"
		signal = 0.99 * signal / signal.abs().max()
		with torch.no_grad():
			center = self.net_z(signal.unsqueeze(0))
		center = center / center.norm()
		return center

	def convert_loudness_to_norm(self, loudness: float = None):
		if loudness is None:
			loudness = self.loudness
		norm = (10 ** (loudness / 20.0) - self._min_norm) * math.sqrt(self.net_g.nz)
		return norm

	def convert_norm_to_loudness(self, norm: float):
		loudness = 20 * math.log10(norm / self.net_g.nz + self._min_norm)
		return loudness

	def update_history(self):
		members = {'center',
				   'x',
				   'y',
				   'angle_x',
				   'angle_y',
				   'angle_max',
				   'loudness'}
		status = {}
		for m in members:
			status[m] = getattr(self, m)
		self.history.append(status)
		self.history = self.history[-self.max_history_len:]
		if self.history_path is not None:
			torch.save(self.history, self.history_path)


def spherical_interpolation(x: torch.Tensor, y: torch.Tensor, t: float):
	theta = torch.acos(torch.dot(x.view(-1), y.view(-1)) / (x.norm() * y.norm()))
	x_t = (torch.sin((1 - t) * theta) * x + torch.sin(t * theta) * y) / torch.sin(theta)
	return x_t


def fading(audio: torch.Tensor):
	sr = 44100.0
	fader = torch.arange(audio.size(-1)).float() / sr
	mu = 0.400
	sigma = 0.030
	fader = 1 - torch.sigmoid((fader - mu) / sigma)

	n = audio.view(-1, audio.size(-1)).size(0)
	fader = torch.stack([fader] * n, dim=0).view_as(audio)
	return audio * fader
