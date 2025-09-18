import matplotlib.pyplot as plt
import librosa
import numpy as np

def calculate_si_sdr(reference, estimate, eps=1e-8):
    # Make sure they are numpy arrays
    if not isinstance(reference, np.ndarray):
        reference = reference.detach().cpu().numpy()
    if not isinstance(estimate, np.ndarray):
        estimate = estimate.detach().cpu().numpy()
    
    # Align lengths
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len]
    estimate = estimate[:min_len]
    
    # Scale-invariant projection
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
    target = alpha * reference
    noise = estimate - target
    
    si_sdr_value = 10 * np.log10(np.sum(target**2) / (np.sum(noise**2) + eps))
    return si_sdr_value

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

        si_sdr = calculate_si_sdr(data_list[0], data_list[i])

        if titles is not None:
            axes[i].set_title(f"{titles[i]}\nSI-SDR={si_sdr:.2f} dB")
        else:
            axes[i].set_title(f"Signal {i+1}\nSI-SDR={si_sdr:.2f} dB")

    fig.colorbar(img, ax=axes, format="%+2.f dB")
    plt.tight_layout()
    plt.show()