import torchaudio
from interface import ImpactDrums


def new_file_received(interface: ImpactDrums, filepath: str):
    # Receive new_path
    signal = torchaudio.load(filepath, normalization=True)
    # Encode waveform
    center = interface.encode_signal(signal)
    # Update plane
    interface.generate_plane(center=center)
    # Decode latent representation
    signal = interface.generate_signal()
    # Write wav file
    torchaudio.save(interface.wav_path, signal, 44100)
    # Return new path
    return interface.wav_path


def generate_sound(
        interface: ImpactDrums,
        angle_x: float,
        angle_y: float,
        angle_max: float,
        loudness: float):
    # Update Parameters
    interface.change_angle_max(angle_max)
    interface.change_angles(angle_x, angle_y)
    interface.change_loudness(loudness)
    # Generate waveform given parameters
    signal = interface.generate_signal()
    # Write wav file
    torchaudio.save(interface.wav_path, signal, 44100)
    # Return new path
    return interface.wav_path


def new_plane(
        interface: ImpactDrums,
        angle_x: float,
        angle_y: float):
    # Generate new_plane while keeping the same center
    interface.generate_plane(center=interface.center)
    # Update angles
    interface.change_angles(angle_x, angle_y)
    # Generate signal given angle_x and angle_y in the new plane
    signal = interface.generate_signal()
    # Write wav file
    torchaudio.save(interface.wav_path, signal, 44100)
    # Return new path
    return interface.wav_path


def new_center(interface: ImpactDrums):
    # Generate new plane with a new center
    interface.generate_plane()
    # Generate signal given angle_x and angle_y in the new plane
    signal = interface.generate_signal()
    # Write wav file
    torchaudio.save(interface.wav_path, signal, 44100)
    # Return new path
    return interface.wav_path


def recenter(
        interface: ImpactDrums,
        angle_x: float,
        angle_y: float):
    # Recenter plan around given point
    interface.change_angles(angle_x, angle_y)
    interface.recenter()
    # Generate signal
    signal = interface.generate_signal()
    # Write wav file
    torchaudio.save(interface.wav_path, signal, 44100)
    # Return new path
    return interface.wav_path


def change_range(
        interface: ImpactDrums,
        angle_max: float):
    # Update angle_max
    interface.change_angle_max(angle_max)
    # Generate new signal
    signal = interface.generate_signal()
    # Write new signal
    torchaudio.save(interface.wav_path, signal, 44100)
    # Return new path
    return interface.wav_path
