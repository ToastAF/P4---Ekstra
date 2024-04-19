import os
import sys
import math
import torch
from torch import nn
import torchaudio
import torchaudio.transforms as tf
from models.utils.io import load_model
import soundfile as sf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

how_many_sounds = 1


def fading(audio: torch.Tensor):
    sr = 44100.0
    fader = torch.arange(audio.size(-1)).float() / sr
    mu = 0.400
    sigma = 0.030
    fader = 1 - torch.sigmoid((fader - mu) / sigma)

    n = audio.view(-1, audio.size(-1)).size(0)
    fader = torch.stack([fader] * n, dim=0).view_as(audio)
    return audio * fader


class ImpactDrums:
    def __init__(self,
                 net_g: nn.Module = r'pretrained\kick_g'
                    ):
    #             net_z: nn.Module = 'pretrained/kick_z'):

        if isinstance(net_g, str):
            net_g = load_model(net_g)
            net_g.eval()
    #    if isinstance(net_z, str):
    #        net_z = load_model(net_z)
    #        net_z.eval()

        self.net_g = net_g
    #    self.net_z = net_z

    vector = [[[2.3638],
              [0.9938],
              [0.2829],
              [1.4742],
              [0.6134],
              [0.7622],
              [-0.4505],
              [-0.6211],
              [1.0030],
              [0.9797],
              [-1.7293],
              [-1.8210],
              [1.0341],
              [1.7408],
              [1.1445],
              [-0.1078],
              [0.1795],
              [0.7574],
              [0.7564],
              [0.9109]]]

    new_z = torch.tensor(vector).float()

    x = new_z[0][0] #These yoink the numbers from the new_z vector
    y = new_z[0][1]
    #print(x[0])
    #print(y[0])

    def generate_sound(self, z: torch.Tensor = None, ):
        if z is None:
            # z = self.net_g.make_inputs(how_many_sounds)
            z = torch.tensor(self.new_z).float()
            z_temp = z.numpy()
            z_numpy = z_temp[:, :, 0]
            np.savetxt('bruh.txt', z_numpy, fmt='%.5f')
            # print(z_numpy.shape)
            # print(z)

        with torch.no_grad():
            signal = self.net_g(z).squeeze(0)
        signal = fading(signal)
        return signal


if __name__ == "__main__":
    impact = ImpactDrums()
    sound = impact.generate_sound().squeeze().numpy()
    #for i in range(how_many_sounds):

    #sf.write('Lyd 1' + '.wav', sound, 44100)
