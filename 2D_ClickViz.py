import os
import librosa
import librosa.feature
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.interpolate import griddata
import API

generator = API.ImpactDrums()

# Initial sound generation
sound = generator.generate_sound().squeeze().numpy()
sf.write('Lyd 1.wav', sound, 44100)

# Variables
plot_size = 6
pca = PCA(n_components=1)  # PCA to reduce to one principal component

def update_plot(feature):
    sorting1_list = []  # Store feature data for the sorting 1 feature
    sorting2_list = []  # Store feature data for the sorting 2 feature
    feature_values = []
    for x, y in zip(grid_x, grid_y):
        generator.new_z[0][0], generator.new_z[0][1] = x, y
        new_sound = generator.generate_sound().squeeze().numpy()
        sf.write('temp_sound.wav', new_sound, 44100)
        y_audio, sr_audio = librosa.load('temp_sound.wav', sr=None)

        envelope = librosa.onset.onset_strength(y=y_audio, sr=sr_audio)
        temporal_centroid = np.sum(np.arange(len(envelope)) * envelope) / np.sum(envelope)

        if feature == 'Sorting 1':
            sorting1_list.append([librosa.feature.spectral_bandwidth(y=y_audio, sr=sr_audio).mean(), librosa.feature.zero_crossing_rate(y_audio).mean(), librosa.feature.spectral_flatness(y=y_audio).mean()])
            feature_values = pca.fit_transform(sorting1_list).flatten()
        elif feature == 'Sorting 2':
            sorting2_list.append([librosa.feature.spectral_contrast(y=y_audio, sr=sr_audio).mean(axis=1).mean(), temporal_centroid, librosa.feature.rms(y=y_audio).mean()])
            feature_values = pca.fit_transform(sorting2_list).flatten()
        else:
            feature_values = [0]

    normalized_values = (np.array(feature_values) - np.min(feature_values)) / (np.max(feature_values) - np.min(feature_values))
    zi = griddata((np.array(grid_x), np.array(grid_y)), normalized_values, (xi, yi), method='cubic')
    ax.clear()
    ax.scatter(xi, yi, c=zi, cmap='viridis', alpha=0.5, zorder=1)

    x_values = [-2.99,
-2.65,
-0.57,
8.11,
-3.61,
-4,
-2.48,
-3.1,
-3.98,
-3.03,
7.94,
-3.64,
-3.06,
-2.86,
-3.81,
-3.2]
    y_values = [6.74,
4.34,
4.61,
4.34,
5.33,
6,
5.88,
5.3,
7.39,
6.29,
6.7,
4.61,
4.2,
4.82,
7.05,
4.68]
    ax.scatter(x_values, y_values, c='red', zorder=2)

    x_right = -2.89
    y_right = 5.19
    ax.scatter(x_right, y_right, c='black', zorder=3)

    canvas.draw()

point = [float(generator.x[0]), float(generator.y[0])]
fig, ax = plt.subplots(figsize=(5, 5))
plt.xlim(generator.x[0] - plot_size, generator.x[0] + plot_size)
plt.ylim(generator.y[0] - plot_size, generator.y[0] + plot_size)
grid_x, grid_y = [], []
offset = plot_size
xi, yi = np.meshgrid(np.linspace(point[0]-offset, point[0]+offset, 100), np.linspace(point[1]-offset, point[1]+offset, 100))
offset_increment = (plot_size*2)/10
for i in range(11):
    for j in range(11):
        grid_x.append((point[0] + i * offset_increment) - offset)
        grid_y.append((point[1] + j * offset_increment) - offset)



window = tk.Tk()
window.title("Generate new sound")
plot_frame = tk.Frame(window)
plot_frame.pack(side=tk.LEFT, padx=10, pady=10)
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.draw()
canvas.get_tk_widget().pack()
right_frame = tk.Frame(window)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

coords_label = tk.Label(right_frame, text="", justify='center')
coords_label.pack()

feature_list = ['Sorting 1', 'Sorting 2']
feature_selector = ttk.Combobox(right_frame, values=feature_list)
feature_selector.pack()
feature_selector.bind("<<ComboboxSelected>>", lambda event: update_plot(feature_selector.get()))

update_plot('Sorting 1')  # Start with a default feature
feature_selector.current(0)
window.mainloop()