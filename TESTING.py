import numpy as np
import librosa
import librosa.feature
import soundfile as sf
from sklearn.decomposition import PCA

import API_Jakob

bruh = API_Jakob.ImpactDrums()

x_line = np.linspace(-5, 5, 100)
feature_values1 = []
feature_values2 = []

pca = PCA(n_components=1)

pca_values1 = []
pca_values2 = []

for x in x_line:
    bruh.new_z[0][bruh.val_1] = x
    new_sound = bruh.generate_sound().squeeze().numpy()
    sf.write('temp_sound.wav', new_sound, 44100)
    y_audio, sr_audio = librosa.load('temp_sound.wav', sr=None)

    bandwidth = librosa.feature.spectral_bandwidth(y=y_audio, sr=sr_audio).mean()
    zcr = librosa.feature.zero_crossing_rate(y_audio).mean()
    flatness = librosa.feature.spectral_flatness(y=y_audio).mean()

    feature_value = np.log1p(bandwidth + zcr + flatness)

    feature_values1.append(feature_value)
    pca_values1.append(pca.fit_transform(np.array(feature_values1).reshape(-1, 1)).flatten())

    feature_values2.append([librosa.feature.spectral_bandwidth(y=y_audio, sr=sr_audio).mean(), librosa.feature.zero_crossing_rate(y_audio).mean(), librosa.feature.spectral_flatness(y=y_audio).mean()])
    pca_values2.append(pca.fit_transform(feature_values2).flatten())

#print(pca_values1)
#print(pca_values2)
