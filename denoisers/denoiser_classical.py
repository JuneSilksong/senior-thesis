import numpy as np
from scipy.signal import stft, istft, medfilt, wiener
import scipy.signal as signal

def spectral_subtraction(x, sr, n_fft=1024, hop=512, win_len_seconds=10, alpha=1.0, beta=0.01):  

    # STFT
    f, t, Zxx = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    n_freq, n_frames = mag.shape
    noise_mag = np.zeros_like(mag)
    
    win_len_frames = win_len_seconds*(sr//hop)

    # Noise estimation via MMSE
    for i in range(n_frames):
        start = max(0, i - win_len_frames)
        noise_mag[:, i] = np.min(mag[:, start:i+1], axis=1)

    # Spectral subtraction
    mag_clean = np.maximum(mag - alpha * noise_mag, beta * noise_mag)

    # Recombine with phase
    Zxx_clean = mag_clean * np.exp(1j * phase)
    Zxx_noise = noise_mag * np.exp(1j * phase) # temp

    # ISTFT
    _, x_clean = istft(Zxx_clean, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return x_clean

def wiener_filter(x, sr, n_fft=1024, hop=512, smoothing=0.98):

    # STFT
    f, t, Zxx = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    # Estimate Pnn
    Pnn = np.mean(mag[:, :10]**2, axis=1, keepdims=True)
    Pyy = np.zeros_like(Pnn)

    # Estimate Pxx and calculate H
    H = np.zeros_like(mag)
    for i in range(mag.shape[1]):
        Pyy = smoothing * Pyy + (1-smoothing) * (mag[:, [i]]**2)
        Pxx_est = Pyy - Pnn
        snr = np.maximum(Pxx_est, 0) / (Pnn + 1e-10)
        H[:, i:i+1] = snr / (1 + snr)

    # Apply filter
    Zxx_clean = H * Zxx

    # ISTFT
    _, x_clean = istft(Zxx_clean, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)

    return x_clean

def kalman_filter(z, R=0.01, Q=1e-5):
    
    # Initial conditions
    n = len(z)
    x_hat = np.zeros(n)
    P = np.zeros(n)
    x_hat[0] = z[0]
    P[0] = 1.0

    for k in range(1, n):
        # Project ahead
        x_hat_minus = x_hat[k-1]
        P_minus = P[k-1] + Q

        # Compute Kalman gain
        K = P_minus / (P_minus + R)

        # Update estimate
        x_hat[k] = x_hat_minus + K * (z[k] - x_hat_minus)

        # Compute error covariance for updated estimate
        P[k] = (1 - K) * P_minus

    return x_hat