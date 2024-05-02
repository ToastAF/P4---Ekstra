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
import API_Jakob

# Create an instance of ImpactDrums
bruh = API_Jakob.ImpactDrums()

# Initial sound generation
sound = bruh.generate_sound().squeeze().numpy()
sf.write('Lyd 1.wav', sound, 44100)

# Variables
plot_size = 5
pca = PCA(n_components=1)  # PCA to reduce dimensions to one principal component

click_history = []


# Function to play closest sound
def play_sound(event):
    x = event.xdata
    if x is not None:
        bruh.new_z[0][bruh.val_1] = x
        click_history.append(x)
        new_sound = bruh.generate_sound().squeeze().numpy()
        sf.write('temp_sound.wav', new_sound, 44100)
        y_audio, sr_audio = librosa.load('temp_sound.wav', sr=None)
        sd.play(y_audio, sr_audio)
        sd.wait()
        coords_label.config(text=f"Clicked coordinates: {x:.2f}")


# Function to update the gradient based on selected feature
def update_plot(feature):
    x_line = np.linspace(-plot_size, plot_size, 100)
    feature_values = []

    for x in x_line:
        bruh.new_z[0][bruh.val_1] = x
        new_sound = bruh.generate_sound().squeeze().numpy()
        sf.write('temp_sound.wav', new_sound, 44100)
        y_audio, sr_audio = librosa.load('temp_sound.wav', sr=None)
        feature_values.append(extract_features(y_audio, sr_audio, feature))

    normalized_values = normalize_features(feature_values)
    ax.clear()
    ax.scatter(x_line, np.zeros_like(x_line), c=normalized_values, cmap='viridis', s=100)
    canvas.draw()


def normalize_features(features):
    features = np.array(features)
    return (features - np.min(features)) / (np.max(features) - np.min(features))


def extract_features(y_audio, sr_audio, feature):
    feature_dict = {
        'Spectral Rolloff': librosa.feature.spectral_rolloff(y=y_audio, sr=sr_audio).mean(),
        'Spectral Contrast': librosa.feature.spectral_contrast(y=y_audio, sr=sr_audio).mean(axis=1).mean(),
        'Spectral Centroid': librosa.feature.spectral_centroid(y=y_audio, sr=sr_audio).mean(),
        'Zero Crossing Rate': librosa.feature.zero_crossing_rate(y_audio).mean(),
        'Spectral Bandwidth': librosa.feature.spectral_bandwidth(y=y_audio, sr=sr_audio).mean(),
        'Spectral Flatness': librosa.feature.spectral_flatness(y=y_audio).mean(),
        'Temporal Centroid': np.sum(np.arange(len(librosa.onset.onset_strength(y=y_audio, sr=sr_audio))) * librosa.onset.onset_strength(y=y_audio, sr=sr_audio)) / np.sum(librosa.onset.onset_strength(y=y_audio, sr=sr_audio)),
        'RMS Energy': librosa.feature.rms(y=y_audio).mean(),
        'Distortion': np.log1p(librosa.feature.spectral_bandwidth(y=y_audio, sr=sr_audio).mean() + librosa.feature.zero_crossing_rate(y_audio).mean() + librosa.feature.spectral_flatness(y=y_audio).mean()),
        'Jakobs mor': librosa.feature.spectral_contrast(y=y_audio, sr=sr_audio).mean(axis=1).mean() + np.sum(np.arange(len(librosa.onset.onset_strength(y=y_audio, sr=sr_audio))) * librosa.onset.onset_strength(y=y_audio, sr=sr_audio)) / np.sum(librosa.onset.onset_strength(y=y_audio, sr=sr_audio)) + librosa.feature.rms(y=y_audio).mean()
    }
    return feature_dict.get(feature, 0)


# Initialize plot and GUI components
fig, ax = plt.subplots(figsize=(10, 2))
ax.set_xlim(-plot_size, plot_size)
ax.set_ylim(0, 1)  # Fixed Y-axis to make it look more like a 1D line

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

feature_list = ['Spectral Rolloff', 'Spectral Contrast', 'Spectral Centroid', 'Zero Crossing Rate', 'Spectral Bandwidth', 'Spectral Flatness', 'Temporal Centroid', 'RMS Energy', 'Distortion', 'Jakobs mor']
feature_selector = ttk.Combobox(right_frame, values=feature_list)
feature_selector.pack()
feature_selector.bind("<<ComboboxSelected>>", lambda event: update_plot(feature_selector.get()))

fig.canvas.mpl_connect('button_press_event', play_sound)
update_plot('Spectral Rolloff')  # Start with a default feature
feature_selector.current(0)

window.mainloop()

#Vi gemmer en fil med alle de ting som brugeren har klikket på
np.savetxt('1D_click_history.txt', click_history, fmt='%.5f')
