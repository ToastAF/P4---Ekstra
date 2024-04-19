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
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import numpy as np

how_many_sounds = 500

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
                 net_g: nn.Module = r'C:\Users\jakob\Desktop\ImpactDrumsAPI\pretrained\kick_g'
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
    def generate_sound(self, z: torch.Tensor = None):
        if z is None:
            z = self.net_g.make_inputs(how_many_sounds)
            z_temp = z.numpy()
            z_numpy = z_temp[:, :, 0]
            np.savetxt('bruh.txt', z_numpy, fmt='%.5f')
            # print(z_numpy.shape)
            # print(z[0])

            pca = PCA(n_components=2, random_state=42)
            reduced_features = pca.fit_transform(z_numpy)
            dbscan = DBSCAN(eps=0.05, min_samples=5)
            labels = dbscan.fit_predict(reduced_features)

            plt.figure(figsize=(10, 7))
            plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', marker='o')
            plt.title('PCA plot of 100 sounds')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.grid(True)
            plt.colorbar(label='Cluster')
            plt.show()

        with torch.no_grad():
            signal = self.net_g(z).squeeze(0)
        signal = fading(signal)
        return signal

if __name__ == "__main__":
    impact = ImpactDrums()
    sound = impact.generate_sound().squeeze().numpy()
    #for i in range(how_many_sounds):
    #    sf.write('test' + str(i) + '.wav', sound[i], 44100)
