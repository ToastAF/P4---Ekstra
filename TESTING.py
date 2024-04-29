import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable
from scipy.interpolate import griddata
import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import soundfile as sf
import librosa
import librosa.feature
import API_Jakob

# Create an instance of ImpactDrums, which is the API, that generates sounds
generator = API_Jakob.ImpactDrums()

# Initial sound generation
sound = generator.generate_sound().squeeze().numpy()
sf.write('Lyd 1.wav', sound, 44100)


def update_plot(ax, canvas, feature, grid_x, grid_y, xi, yi, pca):
    my_feature_list = []  # Store feature data for MyFeature
    distortion_feature_list = []  # Store feature data for Jakob
    feature_list = []  # Store feature data for Bruh
    spectral_feature_list = []  # Store feature data for Spectral stuff
    feature_values = []
    for x, y in zip(grid_x, grid_y):
        generator.new_z[0][0], generator.new_z[0][1] = x, y
        new_sound = generator.generate_sound().squeeze().numpy()
        sf.write('temp_sound.wav', new_sound, 44100)
        y_audio, sr_audio = librosa.load('temp_sound.wav', sr=None)

        envelope = librosa.onset.onset_strength(y=y_audio, sr=sr_audio)
        temporal_centroid = np.sum(np.arange(len(envelope)) * envelope) / np.sum(envelope)

        if feature == 'MyFeature':
            # MyFeature is a combination of Spectral Centroid, RMS Energy and Temporal Centroid
            my_feature_list.append([librosa.feature.spectral_centroid(y=y_audio, sr=sr_audio).mean(), librosa.feature.rms(y=y_audio).mean(), temporal_centroid])
            feature_values = pca.fit_transform(my_feature_list).flatten()
        elif feature == 'Distortion':
            distortion_feature_list.append([librosa.feature.spectral_bandwidth(y=y_audio, sr=sr_audio).mean(), librosa.feature.zero_crossing_rate(y_audio).mean(), librosa.feature.spectral_flatness(y=y_audio).mean()])
            feature_values = pca.fit_transform(distortion_feature_list).flatten()
        elif feature == 'Bruh':
            # Bruh is a combination of Spectral Bandwidth, Zero Crossing Rate and Spectral Rolloff
            feature_list.append([librosa.feature.spectral_bandwidth(y=y_audio, sr=sr_audio).mean(), librosa.feature.zero_crossing_rate(y_audio).mean(), librosa.feature.spectral_rolloff(y=y_audio).mean(), librosa.feature.rms(y=y_audio).mean()])
            feature_values = pca.fit_transform(feature_list).flatten()
        elif feature == 'Spectral':
            # Bruh is a combination of Spectral Bandwidth, Zero Crossing Rate and Spectral Rolloff
            feature_list.append([librosa.feature.spectral_bandwidth(y=y_audio, sr=sr_audio).mean(), librosa.feature.spectral_contrast(y=y_audio).mean(), librosa.feature.spectral_rolloff(y=y_audio).mean(), librosa.feature.spectral_centroid(), temporal_centroid])
            feature_values = pca.fit_transform(feature_list).flatten()
        else:
            # Using computed values directly
            value = {
                'Spectral Rolloff': librosa.feature.spectral_rolloff(y=y_audio, sr=sr_audio).mean(),
                'Spectral Contrast': librosa.feature.spectral_contrast(y=y_audio, sr=sr_audio).mean(axis=1).mean(),
                'Spectral Centroid': librosa.feature.spectral_centroid(y=y_audio, sr=sr_audio).mean(),
                'Zero Crossing Rate': librosa.feature.zero_crossing_rate(y_audio).mean(),
                'Spectral Bandwidth': librosa.feature.spectral_bandwidth(y=y_audio, sr=sr_audio).mean(),
                'Spectral Flatness': librosa.feature.spectral_flatness(y=y_audio).mean(),
                'Temporal Centroid': temporal_centroid,
                'RMS Energy': librosa.feature.rms(y=y_audio).mean()
            }.get(feature)
            feature_values.append(value)

    normalized_values = (np.array(feature_values) - np.min(feature_values)) / (np.max(feature_values) - np.min(feature_values))
    zi = griddata((np.array(grid_x), np.array(grid_y)), normalized_values, (xi, yi), method='cubic')

    ax.clear()
    scatter = ax.scatter(xi, yi, c=zi, cmap='viridis', alpha=0.5)

    # Update color bar based on the scatter plot's data
    norm = Normalize(vmin=np.min(normalized_values), vmax=np.max(normalized_values))
    mappable = ScalarMappable(norm=norm, cmap='viridis')
    mappable.set_array([])  # Set dummy array for the color bar
    cbar = ColorbarBase(ax.figure.add_axes([0.85, 0.1, 0.05, 0.8]), cmap='viridis', norm=norm)

    cbar.ax.yaxis.tick_left()  # Position color bar ticks on the left

    # Redraw canvas
    canvas.draw()


# Usage example:
fig, ax = plt.subplots()
canvas = ax.figure.canvas
feature = 'MyFeature'  # Example feature name
update_plot(ax, canvas, feature, grid_x, grid_y, xi, yi, pca)
plt.show()
