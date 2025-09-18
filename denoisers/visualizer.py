import matplotlib.pyplot as plt
import librosa
import numpy as np

def visualize_noise(data, rate):
  D = librosa.stft(data)
  librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max), sr=rate, y_axis='log', x_axis='time')
  plt.colorbar()
  plt.show()

def visualize_multiple_noises(data_list, rate, titles=None):
  n = len(data_list)
  fig, axes = plt.subplots(1, n, figsize=(min(5*n,15), 4), sharey=True)

  if n == 1:
      axes = [axes]

  for i, data in enumerate(data_list):
      D = librosa.stft(data)
      S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
      img = librosa.display.specshow(
          S_db, sr=rate, y_axis='log', x_axis='time', ax=axes[i]
      )
      if titles is not None:
          axes[i].set_title(titles[i])
      else:
          axes[i].set_title(f"Signal {i+1}")

  fig.colorbar(img, ax=axes, format="%+2.f dB")
  plt.tight_layout()
  plt.show()