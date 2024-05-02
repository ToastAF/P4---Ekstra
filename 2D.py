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

# Create an instance of ImpactDrums, which is the API, that generates sounds
generator = API.ImpactDrums()

# Initial sound generation
sound = generator.generate_sound().squeeze().numpy()
sf.write('Lyd 1.wav', sound, 44100)

# Variables
plot_size = 6
pca = PCA(n_components=1)  # PCA to reduce to one principal component

click_history = []

# Function to play the sound associated with the point clicked
def play_sound(event):
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        if isinstance(x, np.ndarray):
            x = x[0]
        if isinstance(y, np.ndarray):
            y = y[0]
        x, y = float(x), float(y)

        click_history.append([x, y])
        generator.new_z[0][0], generator.new_z[0][1] = x, y
        new_sound = generator.generate_sound().squeeze().numpy()
        sf.write('temp_sound.wav', new_sound, 44100)
        y_audio, sr_audio = librosa.load('temp_sound.wav', sr=None)
        sd.play(y_audio, sr_audio)
        sd.wait()
        coords_label.config(text=f"Clicked coordinates: {x:.2f}, {y:.2f}")

# Function to update the gradient based on selected feature
def update_plot(feature):
    MyFeatureList = []  # Store feature data for MyFeature
    DistortionFeatureList = []  # Store feature data for Jakob
    FeatureList = []  # Store feature data for Bruh
    SpectralFeatureList = []  # Store feature data for Spectral stuff
    feature_values = []
    for x, y in zip(grid_x, grid_y):
        generator.new_z[0][0], generator.new_z[0][1] = x, y
        new_sound = generator.generate_sound().squeeze().numpy()
        sf.write('temp_sound.wav', new_sound, 44100)
        y_audio, sr_audio = librosa.load('temp_sound.wav', sr=None)

        envelope = librosa.onset.onset_strength(y=y_audio, sr=sr_audio)
        temporal_centroid = np.sum(np.arange(len(envelope)) * envelope) / np.sum(envelope)

        if feature == 'Sorting 1':
            DistortionFeatureList.append([librosa.feature.spectral_bandwidth(y=y_audio, sr=sr_audio).mean(), librosa.feature.zero_crossing_rate(y_audio).mean(), librosa.feature.spectral_flatness(y=y_audio).mean()])
            feature_values = pca.fit_transform(DistortionFeatureList).flatten()
        elif feature == 'Sorting 2':
            FeatureList.append([librosa.feature.spectral_contrast(y=y_audio, sr=sr_audio).mean(axis=1).mean(), temporal_centroid, librosa.feature.rms(y=y_audio).mean()])
            feature_values = pca.fit_transform(FeatureList).flatten()
        else:
            feature_values = [0]

    normalized_values = (np.array(feature_values) - np.min(feature_values)) / (np.max(feature_values) - np.min(feature_values))
    zi = griddata((np.array(grid_x), np.array(grid_y)), normalized_values, (xi, yi), method='cubic')
    ax.clear()
    ax.scatter(xi, yi, c=zi, cmap='viridis', alpha=0.5)
    canvas.draw()

# Initialize plot and GUI
point = [float(generator.x[0]), float(generator.y[0])]
fig, ax = plt.subplots(figsize=(5, 5))
plt.xlim(generator.x[0] - plot_size, generator.x[0] + plot_size)
plt.ylim(generator.y[0] - plot_size, generator.y[0] + plot_size)
grid_x, grid_y = [], []
offset = plot_size
offset_increment = (plot_size*2)/10
for i in range(11):
    for j in range(11):
        grid_x.append((point[0] + i * offset_increment) - offset)
        grid_y.append((point[1] + j * offset_increment) - offset)
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

feature_list = ['Sorting 1', 'Sorting 2']
feature_selector = ttk.Combobox(right_frame, values=feature_list)
feature_selector.pack()
feature_selector.bind("<<ComboboxSelected>>", lambda event: update_plot(feature_selector.get()))

fig.canvas.mpl_connect('button_press_event', play_sound)
update_plot('Sorting 1')  # Start with a default feature
feature_selector.current(0)
window.mainloop()

#Vi gemmer en fil med alle de ting som brugeren har klikket på
np.savetxt('2D_click_history.txt', click_history, fmt='%.5f')
