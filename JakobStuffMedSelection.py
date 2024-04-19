import os
import librosa
import librosa.feature
import pandas as pd
import numpy as np
import mplcursors
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.interpolate import griddata

import API_Jakob

# Create an instance of ImpactDrums
bruh = API_Jakob.ImpactDrums()

# Initial sound generation
sound = bruh.generate_sound().squeeze().numpy()
sf.write('Lyd 1.wav', sound, 44100)

# Variables
plot_size = 5

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Function to play closest sound
def play_closest_sound(event):
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        if isinstance(x, np.ndarray):
            x = x[0]
        if isinstance(y, np.ndarray):
            y = y[0]

        x, y = float(x), float(y)
        bruh.new_z[0][0], bruh.new_z[0][1] = x, y
        new_sound = bruh.generate_sound().squeeze().numpy()
        sf.write('temp_sound.wav', new_sound, 44100)
        file_path = 'temp_sound.wav'
        y_audio, sr_audio = librosa.load(file_path, sr=None)

        sd.play(y_audio, sr_audio)
        sd.wait()

        clicked_point = np.array([x, y])
        coords_label.config(text=f"Clicked coordinates: {x:.2f}, {y:.2f}")

# Function to update the gradient based on selected feature
def update_plot(feature):
    feature_values = []
    for x, y in zip(grid_x, grid_y):
        bruh.new_z[0][0], bruh.new_z[0][1] = x, y
        new_sound = bruh.generate_sound().squeeze().numpy()
        sf.write('temp_sound.wav', new_sound, 44100)
        file_path = 'temp_sound.wav'
        y_audio, sr_audio = librosa.load(file_path, sr=None)

        if feature == 'Spectral Rolloff':
            value = librosa.feature.spectral_rolloff(y=y_audio, sr=sr_audio).mean()
        elif feature == 'Spectral Contrast':
            value = librosa.feature.spectral_contrast(y=y_audio, sr=sr_audio).mean(axis=1).mean()
        elif feature == 'Spectral Centroid':
            value = librosa.feature.spectral_centroid(y=y_audio, sr=sr_audio).mean()
        elif feature == 'Zero Crossing Rate':
            value = librosa.feature.zero_crossing_rate(y_audio).mean()
        elif feature == 'MFCC':
            mfccs = librosa.feature.mfcc(y=y_audio, sr=sr_audio)
            value = np.mean(mfccs[0:5])  # Averaging first 5 MFCCs
        elif feature == 'Spectral Bandwidth':
            value = librosa.feature.spectral_bandwidth(y=y_audio, sr=sr_audio).mean()
        elif feature == 'Spectral Flatness':
            value = librosa.feature.spectral_flatness(y=y_audio).mean()
        elif feature == 'Temporal Centroid':
            envelope = librosa.onset.onset_strength(y=y_audio, sr=sr_audio)
            frames = np.arange(len(envelope))
            temporal_centroid = np.sum(frames * envelope) / np.sum(envelope)
            value = temporal_centroid / len(envelope)
        elif feature == 'RMS Energy':
            value = librosa.feature.rms(y=y_audio).mean()
        feature_values.append(value)

    normalized_values = (np.array(feature_values) - np.min(feature_values)) / (
                np.max(feature_values) - np.min(feature_values))
    zi = griddata((np.array(grid_x), np.array(grid_y)), normalized_values, (xi, yi), method='cubic')
    ax.clear()
    ax.scatter(xi, yi, c=zi, cmap='viridis', alpha=0.5)
    canvas.draw()

# Initialize plot and GUI
point = [float(bruh.x[0]), float(bruh.y[0])]
fig, ax = plt.subplots(figsize=(5, 5))
plt.xlim(bruh.x[0]-plot_size, bruh.x[0]+plot_size)
plt.ylim(bruh.y[0]-plot_size, bruh.y[0]+plot_size)
grid_x, grid_y = [], []
offset = (plot_size*2)/10
for i in range(11):
    for j in range(11):
        grid_x.append((point[0] + i * offset)-1)
        grid_y.append((point[1] + j * offset)-1)

xi, yi = np.meshgrid(np.linspace(min(grid_x), max(grid_x), 100), np.linspace(min(grid_y), max(grid_y), 100))

# GUI setup
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

# Update feature list in your GUI
feature_list = ['Spectral Rolloff', 'Spectral Contrast', 'Spectral Centroid', 'Zero Crossing Rate',
                'MFCC', 'Spectral Bandwidth', 'Spectral Flatness', 'Temporal Centroid', 'RMS Energy']
feature_selector = ttk.Combobox(right_frame, values=feature_list)
feature_selector.pack()
feature_selector.bind("<<ComboboxSelected>>", lambda event: update_plot(feature_selector.get()))


fig.canvas.mpl_connect('button_press_event', play_closest_sound)
update_plot('Spectral Rolloff')
window.mainloop()
