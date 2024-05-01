import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from sklearn.decomposition import PCA

# Assuming API_Jakob is correctly configured and available
import API_Jakob

# Initialize the API for generating sounds
bruh = API_Jakob.ImpactDrums()

# Variables
plot_size = 5
pca = PCA(n_components=1)  # PCA for feature reduction


# Function to play sound at a specific position
def play_sound(x):
    bruh.new_z[0][bruh.val_1] = x
    new_sound = bruh.generate_sound().squeeze().numpy()
    sf.write('temp_sound.wav', new_sound, 44100)
    y_audio, sr_audio = librosa.load('temp_sound.wav', sr=None)
    sd.play(y_audio, sr_audio)
    sd.wait()


# Function to update the plot based on the selected feature
def update_plot(feature):
    x_line = np.linspace(-plot_size, plot_size, 100)
    feature_values = []

    # Collect feature data for each point along the line
    for x in x_line:
        bruh.new_z[0][bruh.val_1] = x
        new_sound = bruh.generate_sound().squeeze().numpy()
        sf.write('temp_sound.wav', new_sound, 44100)
        y_audio, sr_audio = librosa.load('temp_sound.wav', sr=None)

        # Select the feature to analyze
        feature_value = {
            'Spectral Centroid': librosa.feature.spectral_centroid(y=y_audio, sr=sr_audio).mean(),
            'Spectral Rolloff': librosa.feature.spectral_rolloff(y=y_audio, sr=sr_audio).mean(),
            # Add other features as necessary
        }.get(feature, 0)
        feature_values.append(feature_value)

    # Normalize the feature values
    normalized_values = (np.array(feature_values) - np.min(feature_values)) / (
                np.max(feature_values) - np.min(feature_values))

    # Apply PCA for dimensionality reduction
    pca_values = pca.fit_transform(np.array(feature_values).reshape(-1, 1)).flatten()
    normalized_pca_values = (pca_values - np.min(pca_values)) / (np.max(pca_values) - np.min(pca_values))

    # Clear and update the plot
    ax.clear()
    ax.scatter(x_line, np.zeros_like(x_line), c=normalized_pca_values, cmap='viridis')
    canvas.draw()


# Set up the GUI
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

feature_selector = ttk.Combobox(right_frame, values=['Spectral Centroid', 'Spectral Rolloff'])
feature_selector.pack()
feature_selector.bind("<<ComboboxSelected>>", lambda event: update_plot(feature_selector.get()))

fig.canvas.mpl_connect('button_press_event', lambda event: play_sound(event.xdata))

update_plot('Spectral Centroid')  # Start with a default feature
feature_selector.current(0)

window.mainloop()
