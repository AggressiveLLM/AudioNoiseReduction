import librosa
import numpy as np
import soundfile as sf
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def wiener_filter(noisy, clean, noise, para):
    n_fft = para["n_fft"]
    hop_length = para["hop_length"]
    win_length = para["win_length"]
    alpha = para["alpha"]
    beta = para["beta"]

    S_noisy = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length)  # DxT
    S_noise = librosa.stft(noise, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    S_clean = librosa.stft(clean, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    Pxx = np.mean((np.abs(S_clean)) ** 2, axis=1, keepdims=True)  # Dx1
    Pnn = np.mean((np.abs(S_noise)) ** 2, axis=1, keepdims=True)

    H = (Pxx / (Pxx + alpha * Pnn)) ** beta

    S_enhec = S_noisy * H

    # 使用length参数确保输出长度与输入相同
    enhenc = librosa.istft(S_enhec, hop_length=hop_length, win_length=win_length, length=len(noisy))

    return H, enhenc, Pxx, Pnn, S_noisy


def compute_snr_improvement(clean, noisy, enhanced):
    """Calculate SNR improvement"""
    # 统一信号长度
    min_len = min(len(clean), len(noisy), len(enhanced))
    clean = clean[:min_len]
    noisy = noisy[:min_len]
    enhanced = enhanced[:min_len]

    # Compute original noise
    noise_original = noisy - clean
    # Compute residual noise after enhancement
    noise_residual = enhanced - clean

    # Avoid division by zero
    eps = 1e-10

    # Original SNR
    signal_power = np.mean(clean ** 2)
    noise_original_power = np.mean(noise_original ** 2)
    original_snr = 10 * np.log10(signal_power / (noise_original_power + eps))

    # Enhanced SNR
    noise_residual_power = np.mean(noise_residual ** 2)
    enhanced_snr = 10 * np.log10(signal_power / (noise_residual_power + eps))

    # SNR improvement
    snr_improvement = enhanced_snr - original_snr

    return original_snr, enhanced_snr, snr_improvement


def compute_spectral_contrast(clean, enhanced, fs, n_fft=256):
    """Calculate spectral contrast"""
    # 统一信号长度
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]

    S_clean = np.abs(librosa.stft(clean, n_fft=n_fft))
    S_enhanced = np.abs(librosa.stft(enhanced, n_fft=n_fft))

    # Compute mean spectrum
    mean_clean = np.mean(S_clean, axis=1)
    mean_enhanced = np.mean(S_enhanced, axis=1)

    # Spectral difference
    spectral_diff = mean_enhanced - mean_clean

    return spectral_diff


def analyze_frequency_bands(fs, H, Pxx, Pnn):
    """Analyze processing effect in different frequency bands"""
    # Frequency bands (Hz)
    bands = {
        'Low (0-300Hz)': (0, 300),
        'Mid-Low (300-1000Hz)': (300, 1000),
        'Mid (1000-3000Hz)': (1000, 3000),
        'Mid-High (3000-6000Hz)': (3000, 6000),
        'High (6000-8000Hz)': (6000, fs / 2)
    }

    # Frequency axis
    freq_bins = np.linspace(0, fs / 2, len(H))

    band_results = []
    for band_name, (f_low, f_high) in bands.items():
        # Create frequency band mask
        mask = (freq_bins >= f_low) & (freq_bins < f_high)

        if np.sum(mask) > 0:  # Ensure there are frequency points
            # Average gain
            avg_gain = np.mean(H[mask].squeeze())
            gain_db = 10 * np.log10(avg_gain + 1e-10)

            # Average SNR
            snr = np.mean(Pxx[mask] / (Pnn[mask] + 1e-10))
            snr_db = 10 * np.log10(snr + 1e-10)

            # Noise suppression ratio
            noise_suppression = 1 - avg_gain

            band_results.append({
                'band': band_name,
                'avg_gain_db': gain_db,
                'snr_db': snr_db,
                'noise_suppression': noise_suppression * 100  # Percentage
            })

    return band_results


def create_comprehensive_filter_plot(fs, H, Pxx, Pnn, spectral_diff, band_results,
                                     original_snr, enhanced_snr, snr_improvement,
                                     save_path="wiener_filter_analysis.png"):
    """Create comprehensive filter analysis plot"""

    fig = plt.figure(figsize=(18, 14))

    # 1. Filter Frequency Response (Main plot)
    ax1 = plt.subplot(3, 3, 1)
    freq_bins = np.linspace(0, fs / 2, len(H))
    H_dB = 10 * np.log10(H.squeeze() + 1e-10)

    plt.plot(freq_bins, H_dB, 'b-', linewidth=2, label='Filter Gain')
    plt.fill_between(freq_bins, H_dB, -60, where=(H_dB > -60),
                     color='blue', alpha=0.1, label='Gain Region')

    # Mark key frequency points
    for freq in [250, 500, 1000, 2000, 4000, 8000]:
        if freq < fs / 2:
            idx = np.argmin(np.abs(freq_bins - freq))
            plt.plot(freq, H_dB[idx], 'ro', markersize=8)
            plt.text(freq, H_dB[idx] + 3, f'{freq}Hz',
                     ha='center', fontsize=9, fontweight='bold')

    plt.title('Wiener Filter Frequency Response', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Gain (dB)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.xlim([0, fs / 2])
    plt.ylim([-60, 5])

    # 2. Power Spectrum Comparison (修复了这里的变量名错误)
    ax2 = plt.subplot(3, 3, 2)
    freq_bins_power = np.linspace(0, fs / 2, len(Pxx))

    Pxx_dB = 10 * np.log10(Pxx.squeeze() + 1e-10)
    Pnn_dB = 10 * np.log10(Pnn.squeeze() + 1e-10)

    plt.plot(freq_bins_power, Pxx_dB, 'g-', linewidth=2, label='Speech Power')
    plt.plot(freq_bins_power, Pnn_dB, 'r-', linewidth=2, label='Noise Power')
    plt.fill_between(freq_bins_power, Pxx_dB, Pnn_dB,  # 这里修正了变量名
                     where=(Pxx_dB > Pnn_dB), color='green', alpha=0.2, label='Speech Dominant')
    plt.fill_between(freq_bins_power, Pxx_dB, Pnn_dB,  # 这里也修正了
                     where=(Pxx_dB <= Pnn_dB), color='red', alpha=0.2, label='Noise Dominant')

    plt.title('Power Spectrum Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Power (dB)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlim([0, fs / 2])

    # 3. Band Analysis Bar Chart
    ax3 = plt.subplot(3, 3, 3)
    bands = [r['band'] for r in band_results]
    gains = [r['avg_gain_db'] for r in band_results]
    suppression = [r['noise_suppression'] for r in band_results]

    x_pos = np.arange(len(bands))
    bars = plt.bar(x_pos, gains, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(bands))))

    # Add value labels
    for i, (bar, gain, sup) in enumerate(zip(bars, gains, suppression)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{gain:.1f}dB\n({sup:.1f}%)', ha='center', va='bottom', fontsize=9)

    plt.title('Frequency Band Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency Band', fontsize=12)
    plt.ylabel('Average Gain (dB)', fontsize=12)
    plt.xticks(x_pos, bands, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    # 4. SNR Performance Metrics
    ax4 = plt.subplot(3, 3, 4)
    categories = ['Original SNR', 'Enhanced SNR', 'SNR Improvement']
    values = [original_snr, enhanced_snr, snr_improvement]
    colors = ['red', 'green', 'blue']

    bars_snr = plt.bar(categories, values, color=colors, alpha=0.7)

    # Add value labels
    for bar, value in zip(bars_snr, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.2,
                 f'{value:.2f}dB', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title('SNR Performance Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('SNR (dB)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')

    # 5. Spectral Difference Analysis
    ax5 = plt.subplot(3, 3, 5)
    freq_bins_diff = np.linspace(0, fs / 2, len(spectral_diff))

    plt.plot(freq_bins_diff, spectral_diff, 'purple', linewidth=2, label='Spectral Difference')
    plt.fill_between(freq_bins_diff, spectral_diff, 0,
                     where=(spectral_diff > 0), color='green', alpha=0.3, label='Enhanced')
    plt.fill_between(freq_bins_diff, spectral_diff, 0,
                     where=(spectral_diff < 0), color='red', alpha=0.3, label='Attenuated')

    # Zero line
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    plt.title('Spectral Difference Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Magnitude Difference', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlim([0, fs / 2])

    # 6. Filter Statistics
    ax6 = plt.subplot(3, 3, 6)
    # Remove axes for text display
    ax6.axis('off')

    # Calculate statistics
    H_linear = H.squeeze()
    avg_gain = np.mean(H_linear)
    std_gain = np.std(H_linear)
    max_gain = np.max(H_linear)
    min_gain = np.min(H_linear)

    # Statistics text
    stats_text = f"""
    Filter Statistics:

    Average Gain: {avg_gain:.4f}
    Max Gain: {max_gain:.4f}
    Min Gain: {min_gain:.4f}
    Gain Std Dev: {std_gain:.4f}

    Frequency Range: 0 - {fs / 2:.0f} Hz
    Frequency Resolution: {fs / 2 / len(H):.1f} Hz/bin
    Frequency Bins: {len(H)}

    Noise Suppression Range: {min(10 * np.log10(H_linear + 1e-10)):.1f} to {max(10 * np.log10(H_linear + 1e-10)):.1f} dB
    """

    plt.text(0.05, 0.95, stats_text, fontsize=10, fontfamily='monospace',
             verticalalignment='top', transform=ax6.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # 7. Gain Distribution Histogram
    ax7 = plt.subplot(3, 3, 7)
    H_dB_linear = 10 * np.log10(H_linear + 1e-10)
    plt.hist(H_dB_linear, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    plt.axvline(x=np.mean(H_dB_linear), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(H_dB_linear):.2f}dB')

    plt.title('Gain Distribution Histogram', fontsize=14, fontweight='bold')
    plt.xlabel('Gain (dB)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 8. Cumulative Gain Curve
    ax8 = plt.subplot(3, 3, 8)
    sorted_H = np.sort(H_linear)
    cumulative = np.cumsum(sorted_H) / np.sum(sorted_H)

    plt.plot(np.arange(len(sorted_H)) / len(sorted_H) * 100, cumulative * 100,
             'b-', linewidth=2)
    plt.fill_between(np.arange(len(sorted_H)) / len(sorted_H) * 100,
                     cumulative * 100, 0, alpha=0.3, color='blue')

    # Mark key points
    percentiles = [25, 50, 75, 90]
    for p in percentiles:
        idx = int(len(sorted_H) * p / 100)
        plt.plot(p, cumulative[idx] * 100, 'ro', markersize=8)
        plt.text(p, cumulative[idx] * 100 + 5, f'{cumulative[idx] * 100:.1f}%',
                 ha='center', fontsize=9)

    plt.title('Cumulative Gain Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency Points (%)', fontsize=12)
    plt.ylabel('Cumulative Gain (%)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 9. Performance Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    # Performance evaluation
    best_band = max(band_results, key=lambda x: x['avg_gain_db'])['band']
    worst_band = min(band_results, key=lambda x: x['avg_gain_db'])['band']

    performance_text = f"""
    Performance Summary:

    SNR Improvement: {snr_improvement:.2f} dB
    {'Excellent (>5dB)' if snr_improvement > 5 else 'Good (2-5dB)' if snr_improvement > 2 else 'Fair (<2dB)'}

    Speech Fidelity: {'High' if np.mean(H_linear) > 0.7 else 'Medium' if np.mean(H_linear) > 0.4 else 'Low'}
    Noise Suppression: {'Strong' if np.mean(H_linear) < 0.3 else 'Medium' if np.mean(H_linear) < 0.6 else 'Weak'}

    Best Band: {best_band}
    Worst Band: {worst_band}

    Recommendations:
    {'Reduce alpha for better speech quality' if np.mean(H_linear) < 0.4 else 'Parameters are well-tuned'}
    """

    plt.text(0.05, 0.95, performance_text, fontsize=10, fontfamily='monospace',
             verticalalignment='top', transform=ax9.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.suptitle('Wiener Filter Comprehensive Analysis Report', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Comprehensive filter analysis saved as: {save_path}")


if __name__ == "__main__":
    # Load clean speech
    clean_wav_file = "clean.wav"
    clean, fs = librosa.load(clean_wav_file, sr=None)

    # Load noisy speech
    noisy_wav_file = "noisy.wav"
    noisy, fs = librosa.load(noisy_wav_file, sr=None)

    # 确保干净语音和噪声语音长度相同
    min_len = min(len(clean), len(noisy))
    clean = clean[:min_len]
    noisy = noisy[:min_len]

    # Get noise
    noise = noisy - clean

    # Set parameters
    para_wiener = {}
    para_wiener["n_fft"] = 256
    para_wiener["hop_length"] = 128
    para_wiener["win_length"] = 256
    para_wiener["alpha"] = 0.4
    para_wiener["beta"] = 8

    # Wiener filtering
    H, enhenc, Pxx, Pnn, S_noisy = wiener_filter(noisy, clean, noise, para_wiener)

    # Save enhanced audio
    sf.write("enhce_1.wav", enhenc, fs)
    print("Enhanced audio saved as: enhce_1.wav")

    # Calculate SNR improvement
    original_snr, enhanced_snr, snr_improvement = compute_snr_improvement(clean, noisy, enhenc)
    print(f"\nSNR Performance:")
    print(f"   Original SNR: {original_snr:.2f} dB")
    print(f"   Enhanced SNR: {enhanced_snr:.2f} dB")
    print(f"   SNR Improvement: {snr_improvement:.2f} dB")

    # Calculate spectral contrast
    spectral_diff = compute_spectral_contrast(clean, enhenc, fs, n_fft=256)

    # Analyze frequency bands
    band_results = analyze_frequency_bands(fs, H, Pxx, Pnn)
    print("\nFrequency Band Analysis:")
    for result in band_results:
        print(f"   {result['band']}: Gain={result['avg_gain_db']:.1f}dB, "
              f"SNR={result['snr_db']:.1f}dB, Noise Suppression={result['noise_suppression']:.1f}%")

    # Create comprehensive filter analysis plot
    create_comprehensive_filter_plot(
        fs, H, Pxx, Pnn, spectral_diff, band_results,
        original_snr, enhanced_snr, snr_improvement,
        save_path="wiener_filter_analysis_1.png"
    )

    # Original spectrogram comparison
    # 确保信号长度一致
    min_len_spec = min(len(clean), len(noisy), len(enhenc))
    clean_spec = clean[:min_len_spec]
    noisy_spec = noisy[:min_len_spec]
    enhenc_spec = enhenc[:min_len_spec]

    fig = plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.specgram(clean_spec, NFFT=256, Fs=fs, cmap='viridis')
    plt.title("Clean Speech Spectrogram", fontsize=12, fontweight='bold')
    plt.ylabel("Frequency (Hz)", fontsize=10)
    plt.xlabel("Time (s)", fontsize=10)

    plt.subplot(3, 1, 2)
    plt.specgram(noisy_spec, NFFT=256, Fs=fs, cmap='viridis')
    plt.title("Noisy Speech Spectrogram", fontsize=12, fontweight='bold')
    plt.ylabel("Frequency (Hz)", fontsize=10)
    plt.xlabel("Time (s)", fontsize=10)

    plt.subplot(3, 1, 3)
    plt.specgram(enhenc_spec, NFFT=256, Fs=fs, cmap='viridis')
    plt.title("Enhanced Speech Spectrogram", fontsize=12, fontweight='bold')
    plt.ylabel("Frequency (Hz)", fontsize=10)
    plt.xlabel("Time (s)", fontsize=10)

    plt.tight_layout()
    plt.savefig("spectrograms_1.png", dpi=150, bbox_inches='tight')
    print("Spectrograms saved as: spectrograms_1.png")

    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Input Files: {clean_wav_file}, {noisy_wav_file}")
    print(f"Output Files:")
    print(f"  • enhce_1.wav (Enhanced audio)")
    print(f"  • spectrograms_1.png (Spectrogram comparison)")
    print(f"  • wiener_filter_analysis_1.png (Comprehensive filter analysis)")
    print("=" * 60)