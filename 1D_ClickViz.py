import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.feature
import sounddevice as sd
import soundfile as sf
from sklearn.decomposition import PCA

# Import the API for sound generation
import API

# Initialize the API for generating sounds
generator = API.ImpactDrums()

# Variables for plot size and PCA configuration
plot_size = 5
pca = PCA(n_components=1)  # Using PCA to reduce the feature dimension to 1D


def update_plot(feature):
    x_line = np.linspace(-plot_size, plot_size, 100)
    feature_values = []

    for x in x_line:
        generator.new_z[0][0] = x
        new_sound = generator.generate_sound().squeeze().numpy()
        sf.write('temp_sound.wav', new_sound, 44100)
        y_audio, sr_audio = librosa.load('temp_sound.wav', sr=None)

        envelope = librosa.onset.onset_strength(y=y_audio, sr=sr_audio)
        temporal_centroid = np.sum(np.arange(len(envelope)) * envelope) / np.sum(envelope)

        # Collect features based on selection
        if feature == 'Sorting 1':
            features = [librosa.feature.spectral_bandwidth(y=y_audio, sr=sr_audio).mean(),
                        librosa.feature.zero_crossing_rate(y_audio).mean(),
                        librosa.feature.spectral_flatness(y=y_audio).mean()]
        elif feature == 'Sorting 2':
            features = [librosa.feature.spectral_contrast(y=y_audio,
                        sr=sr_audio).mean(axis=1).mean(), temporal_centroid,
                        librosa.feature.rms(y=y_audio).mean()]
        else:
            features = [0]  # Default case for unhandled features

        feature_values.append(features)

    # Flatten, normalize, and apply PCA to the collected features
    feature_values = np.array(feature_values)
    pca_values = pca.fit_transform(feature_values).flatten()
    normalized_pca_values = (pca_values - np.min(pca_values)) / (np.max(pca_values) - np.min(pca_values))

    # Update the plot
    ax.clear()
    ax.scatter(x_line, np.zeros_like(x_line), c=normalized_pca_values, cmap='viridis', s=40, zorder=1)  # Visual representation of features

    x_values = [-5,
-4.59,
-4.46,
-3.4,
-4.91,
-5,
-4.95,
-4.02,
-4.79,
-4.18,
4.96,
-5,
-4.56,
4.29,
-5.05,
-5]
    ax.scatter(x_values, np.zeros_like(x_values), c='red', zorder=2)

    x_right = 4.93
    ax.scatter(x_right, 0, c='pink', zorder=3)

    canvas.draw()


window = tk.Tk()
window.title("Sound Feature Visualization")

plot_frame = tk.Frame(window)
plot_frame.pack(side=tk.LEFT, padx=10, pady=10)

fig, ax = plt.subplots(figsize=(10, 2))
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

update_plot('Sorting 1')  # Initialize with a default feature for visualization
feature_selector.current(0)

window.mainloop()
