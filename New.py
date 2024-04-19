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

bruh = API_Jakob.ImpactDrums()
sound = bruh.generate_sound().squeeze().numpy()

sf.write('Lyd 1' + '.wav', sound, 44100)

#Variables
plot_size = 5

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def play_closest_sound(event):
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        if isinstance(x, np.ndarray):
            x = x[0]
        if isinstance(y, np.ndarray):
            y = y[0]

        x = float(x)
        y = float(y)

        bruh.new_z[0][0] = x
        bruh.new_z[0][1] = y
        new_sound = bruh.generate_sound().squeeze().numpy()
        sf.write('temp_sound' + '.wav', new_sound, 44100)
        # print(bruh.new_z)
        file_path = os.path.join(r'C:\Users\jakob\Desktop\ImpactDrumsAPI', 'temp_sound.wav')
        y_audio, sr_audio = librosa.load(file_path, sr=None)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y_audio, sr=sr_audio).mean()
        print(spectral_rolloff)

        sd.play(y_audio, sr_audio)
        sd.wait()

        clicked_point = np.array([x, y])

        coords_label.config(text=f"Clicked coordinates: {x:.2f}, {y:.2f}")


#The plot thickens
point = [float(bruh.x[0]), float(bruh.y[0])]

fig, ax = plt.subplots(figsize=(5, 5))
scatter = ax.scatter(point[0], point[1], zorder=10)

#Limiting the space
plt.xlim(bruh.x[0]-plot_size, bruh.x[0]+plot_size)
plt.ylim(bruh.y[0]-plot_size, bruh.y[0]+plot_size)

#Making the grid
grid = []
feature_grid = []
grid_x = []
grid_y = []
offset = (plot_size*2)/10
for i in range(11):
    for j in range(11):
        grid_x.append((point[0] + i * offset)-1)
        grid_y.append((point[1] + j * offset)-1)
for x, y in zip(grid_x, grid_y):
    grid.append([x, y])

    bruh.new_z[0][0] = x
    bruh.new_z[0][1] = y
    new_sound = bruh.generate_sound().squeeze().numpy()
    sf.write('temp_sound' + '.wav', new_sound, 44100)
    file_path = os.path.join(r'C:\Users\jakob\Desktop\ImpactDrumsAPI', 'temp_sound.wav')
    y_audio, sr_audio = librosa.load(file_path, sr=None)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y_audio, sr=sr_audio).mean()
    spectral_contrast = librosa.feature.spectral_contrast(y=y_audio, sr=sr_audio).mean(axis=1).mean()

    feature_grid.append(spectral_contrast)

#print(grid)
#print(feature_grid)
f_grid_np = np.array(feature_grid)

#The grid
scatter_grid = ax.scatter(grid_x, grid_y, zorder=9)

#The color gradient, hopefully
normalized_values = (f_grid_np - f_grid_np.min()) / (f_grid_np.max() - f_grid_np.min())
xi, yi = np.array(grid_x), np.array(grid_y)
x, y = np.linspace(xi.min(), xi.max(), 100), np.linspace(yi.min(), yi.max(), 100)
xi, yi = np.meshgrid(x, y)
zi = griddata((np.array(grid_x), np.array(grid_y)), normalized_values, (xi, yi), method='cubic')
ax.clear()
scatter = ax.scatter(xi, yi, c=zi, cmap='viridis', alpha=0.5)

#Stuff for the GUI
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

fig.canvas.mpl_connect('button_press_event', play_closest_sound)

window.mainloop()
