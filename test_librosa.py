import librosa
import numpy as np
import matplotlib.pyplot as plt

# Create a simple synthetic audio signal (sine wave at 440 Hz)
sr = 22050  # Sample rate
duration = 2.0  # seconds
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
y = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
print(f'Created synthetic audio with sample rate: {sr}Hz, duration: {duration:.2f}s')

# Test basic librosa functionality
print("Testing pitch extraction...")
f0 = librosa.yin(y, fmin=50, fmax=2000, frame_length=2048, hop_length=512)
print(f'Pitch analysis successful, {len(f0)} frames extracted.')

print("Testing spectral features...")
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
print(f'MFCC extraction successful, shape: {mfcc.shape}')

print("\nLibrosa is working correctly!") 