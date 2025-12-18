import librosa
import numpy as np
import soundfile as sf
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.linalg import norm
import os


def specsub_enhancement(noisy, fs, n_fft=512, hop_length=128, win_length=512,
                        noise_frame_num=50, alpha=2.0, beta=0.1, smooth_win=3):
    """
    优化版谱减法增强函数
    :param noisy: 带噪语音
    :param fs: 采样率
    :param n_fft: FFT点数
    :param hop_length: 步长
    :param win_length: 窗长
    :param noise_frame_num: 噪声帧数量（纯噪声段）
    :param alpha: 过减因子（控制噪声抑制强度，1.0~5.0）
    :param beta: 谱底阈值（避免负谱，0.0~0.5）
    :param smooth_win: 频率平滑窗口大小（抑制音乐噪声）
    :return: 增强后的语音、增强后的幅度谱
    """
    # 1. STFT变换（用汉明窗减少频谱泄漏）
    window = librosa.filters.get_window('hamming', win_length)
    S_noisy = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length,
                           win_length=win_length, window=window)  # D x T
    D, T = np.shape(S_noisy)
    Mag_noisy = np.abs(S_noisy)
    Phase_noisy = np.angle(S_noisy)  # 修复原代码拼写错误：Phase_nosiy -> Phase_noisy
    Power_noisy = Mag_noisy ** 2     # 修复原代码拼写错误：Power_nosiy -> Power_noisy

    # 2. 噪声谱估计（仅用纯噪声段，计算均值+方差，更鲁棒）
    if noise_frame_num >= T:
        noise_frame_num = T // 10  # 防止噪声帧超过总帧数
    # 计算噪声的功率谱（均值+平滑）
    Power_noise = np.mean(Power_noisy[:, :noise_frame_num], axis=1, keepdims=True)
    # 扩展到所有时间帧
    Power_noise = np.tile(Power_noise, [1, T])

    # 3. 优化的谱减操作（过减+谱底阈值+频率平滑）
    # 过减：增强噪声抑制
    Power_sub = Power_noisy - alpha * Power_noise
    # 谱底阈值：避免负谱，抑制音乐噪声
    Power_enhanced = np.maximum(Power_sub, beta * Power_noise)
    # 频率平滑：对相邻频率点做均值平滑，减少频谱波动
    if smooth_win > 1:
        Power_enhanced = np.apply_along_axis(
            lambda x: np.convolve(x, np.ones(smooth_win)/smooth_win, mode='same'),
            axis=0, arr=Power_enhanced
        )
    Mag_enhanced = np.sqrt(Power_enhanced)

    # 4. 重构复数谱并逆变换
    S_enhanced = Mag_enhanced * np.exp(1j * Phase_noisy)
    # 用ISTFT还原，添加长度对齐（避免首尾截断）
    enhanced = librosa.istft(S_enhanced, hop_length=hop_length,
                             win_length=win_length, window=window,
                             length=len(noisy))  # 保证输出长度与输入一致

    return enhanced, Mag_enhanced


def calculate_rms_energy(signal, frame_length=2048, hop_length=512):
    """优化：计算分帧RMS能量，更能反映局部变化"""
    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)
    rms = np.sqrt(np.mean(frames ** 2, axis=0))
    return np.mean(rms)  # 全局均值，也可返回分帧结果


def plot_spectrogram_comparison(noisy, enhanced, output_filename, fs=16000):
    """优化频谱图绘制（匹配STFT参数，保证可视化准确）"""
    # 确保长度一致
    min_len = min(len(noisy), len(enhanced))
    noisy = noisy[:min_len]
    enhanced = enhanced[:min_len]

    # 计算局部能量变化（更能反映增强效果）
    noisy_energy = calculate_rms_energy(noisy)
    enhanced_energy = calculate_rms_energy(enhanced)
    energy_reduction = 10 * np.log10((noisy_energy ** 2 + 1e-7) / (enhanced_energy ** 2 + 1e-7))

    # 统一STFT参数（与增强函数一致）
    n_fft = 512
    hop_length = 128
    win_length = 512
    window = librosa.filters.get_window('hamming', win_length)

    # 创建图形
    fig = plt.figure(figsize=(15, 10))

    # 1. 原始带噪语音频谱（用librosa绘制，匹配STFT参数）
    plt.subplot(3, 1, 1)
    S_noisy = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length,
                           win_length=win_length, window=window)
    db_noisy = librosa.amplitude_to_db(np.abs(S_noisy), ref=np.max)
    img1 = librosa.display.specshow(db_noisy, sr=fs, hop_length=hop_length,
                                    x_axis='time', y_axis='hz', cmap='jet')
    plt.title(f'Noisy Speech Spectrogram (RMS Energy: {noisy_energy:.4f})',
              fontsize=12, fontweight='bold')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(img1, format='%+2.0f dB')

    # 2. SpecSub增强语音频谱
    plt.subplot(3, 1, 2)
    S_enhanced = librosa.stft(enhanced, n_fft=n_fft, hop_length=hop_length,
                              win_length=win_length, window=window)
    db_enhanced = librosa.amplitude_to_db(np.abs(S_enhanced), ref=np.max)
    img2 = librosa.display.specshow(db_enhanced, sr=fs, hop_length=hop_length,
                                    x_axis='time', y_axis='hz', cmap='jet')
    plt.title(f'SpecSub Enhanced Spectrogram (RMS Energy: {enhanced_energy:.4f}, Reduction: {energy_reduction:.2f} dB)',
              fontsize=12, fontweight='bold')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(img2, format='%+2.0f dB')

    # 3. 波形对比图（归一化）
    plt.subplot(3, 1, 3)
    t = np.arange(min_len) / fs
    noisy_norm = noisy / (np.max(np.abs(noisy)) + 1e-7)
    enhanced_norm = enhanced / (np.max(np.abs(enhanced)) + 1e-7)

    plt.plot(t, noisy_norm, 'r-', alpha=0.6, label='Noisy', linewidth=0.8)
    plt.plot(t, enhanced_norm, 'b-', alpha=0.8, label='SpecSub Enhanced', linewidth=1.2)
    plt.title('Waveform Comparison (Normalized)', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, t[-1]])
    plt.ylim([-1.2, 1.2])

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'Spectrogram saved to: {output_filename}')
    print(f'Noisy energy: {noisy_energy:.4f}')
    print(f'Enhanced energy: {enhanced_energy:.4f}')
    print(f'Energy reduction: {energy_reduction:.2f} dB')

    return noisy_energy, enhanced_energy, energy_reduction


def plot_detailed_comparison(noisy, enhanced, output_filename, fs=16000):
    """优化详细对比图（匹配STFT参数）"""
    min_len = min(len(noisy), len(enhanced))
    noisy = noisy[:min_len]
    enhanced = enhanced[:min_len]

    n_fft = 512
    hop_length = 128
    win_length = 512
    window = librosa.filters.get_window('hamming', win_length)

    fig = plt.figure(figsize=(16, 12))

    # 1. 原始带噪频谱
    plt.subplot(3, 2, 1)
    S_noisy = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length,
                           win_length=win_length, window=window)
    db_noisy = librosa.amplitude_to_db(np.abs(S_noisy), ref=np.max)
    librosa.display.specshow(db_noisy, sr=fs, hop_length=hop_length,
                             x_axis='time', y_axis='hz', cmap='jet')
    plt.title('Noisy Speech Spectrogram', fontsize=11)
    plt.ylabel('Frequency (Hz)')

    # 2. 增强后频谱
    plt.subplot(3, 2, 2)
    S_enhanced = librosa.stft(enhanced, n_fft=n_fft, hop_length=hop_length,
                              win_length=win_length, window=window)
    db_enhanced = librosa.amplitude_to_db(np.abs(S_enhanced), ref=np.max)
    librosa.display.specshow(db_enhanced, sr=fs, hop_length=hop_length,
                             x_axis='time', y_axis='hz', cmap='jet')
    plt.title('SpecSub Enhanced Spectrogram', fontsize=11)
    plt.ylabel('Frequency (Hz)')

    # 3. 带噪波形
    plt.subplot(3, 2, 3)
    t = np.arange(min_len) / fs
    plt.plot(t, noisy, 'r-', alpha=0.7, label='Noisy', linewidth=0.8)
    plt.title('Noisy Waveform', fontsize=11)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    # 4. 增强后波形
    plt.subplot(3, 2, 4)
    plt.plot(t, enhanced, 'b-', alpha=0.8, label='Enhanced', linewidth=1.0)
    plt.title('SpecSub Enhanced Waveform', fontsize=11)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    # 5. 频谱包络对比（0-4kHz）
    plt.subplot(3, 2, 5)
    freqs = np.fft.rfftfreq(n_fft, 1/fs)
    noisy_fft = np.abs(librosa.stft(noisy, n_fft=n_fft)[:, 0])  # 取第一帧
    enhanced_fft = np.abs(librosa.stft(enhanced, n_fft=n_fft)[:, 0])
    plt.plot(freqs[:len(freqs)//4], 20*np.log10(noisy_fft[:len(freqs)//4]+1e-7),
             'r-', alpha=0.7, label='Noisy', linewidth=1.0)
    plt.plot(freqs[:len(freqs)//4], 20*np.log10(enhanced_fft[:len(freqs)//4]+1e-7),
             'b-', alpha=0.8, label='Enhanced', linewidth=1.2)
    plt.title('Spectral Envelope (0-4kHz)', fontsize=11)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)

    # 6. 频带能量分布
    plt.subplot(3, 2, 6)
    bands = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
    band_labels = ['0-1k', '1k-2k', '2k-4k', '4k-8k']
    noisy_energy_per_band = []
    enhanced_energy_per_band = []

    # 用STFT计算频带能量（更准确）
    S_noisy_full = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length)
    S_enhanced_full = librosa.stft(enhanced, n_fft=n_fft, hop_length=hop_length)
    freq_bins = librosa.fft_frequencies(sr=fs, n_fft=n_fft)

    for low, high in bands:
        # 找到频带对应的bin索引
        bin_idx = np.where((freq_bins >= low) & (freq_bins <= high))[0]
        if len(bin_idx) == 0:
            noisy_energy_per_band.append(0)
            enhanced_energy_per_band.append(0)
            continue
        # 计算该频带的总能量
        noisy_band_energy = np.sum(np.abs(S_noisy_full[bin_idx, :])**2)
        enhanced_band_energy = np.sum(np.abs(S_enhanced_full[bin_idx, :])**2)
        noisy_energy_per_band.append(noisy_band_energy)
        enhanced_energy_per_band.append(enhanced_band_energy)

    x_pos = np.arange(len(bands))
    width = 0.35
    plt.bar(x_pos - width/2, noisy_energy_per_band, width, alpha=0.7, color='red', label='Noisy')
    plt.bar(x_pos + width/2, enhanced_energy_per_band, width, alpha=0.8, color='blue', label='Enhanced')
    plt.title('Frequency Band Energy Distribution', fontsize=11)
    plt.xlabel('Frequency Bands')
    plt.ylabel('Energy')
    plt.xticks(x_pos, band_labels)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Detailed comparison saved to: {output_filename}')


def main():
    # 创建输出目录
    output_dir = 'specsub_evaluation'
    audio_dir = os.path.join(output_dir, 'audio')
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # 测试文件
    test_noisy_files = [
        '/root/autodl-tmp/Audio Noise Reduction/SPECSUB/noisy.wav',
        # 可添加更多文件
    ]

    results_summary = []

    for noisy_file in test_noisy_files:
        if not os.path.exists(noisy_file):
            print(f"File not found: {noisy_file}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing: {os.path.basename(noisy_file)}")
        print(f"{'=' * 60}")

        # 读取音频（保留原始采样率，避免重采样失真）
        noisy, fs = librosa.load(noisy_file, sr=None)
        print(f"Sample rate: {fs} Hz, Length: {len(noisy)/fs:.2f} s")

        # 运行优化版SpecSub增强
        enhanced, mag_enhanced = specsub_enhancement(
            noisy,
            fs,
            n_fft=512,        # 提高频率分辨率
            hop_length=128,
            win_length=512,
            noise_frame_num=50,  # 调整噪声帧（50帧≈0.4秒，更合理）
            alpha=2.5,        # 过减因子，根据噪声强度调整
            beta=0.15,        # 谱底阈值
            smooth_win=3      # 频率平滑窗口
        )

        # 保存音频
        base_name = os.path.splitext(os.path.basename(noisy_file))[0]
        sf.write(os.path.join(audio_dir, f'{base_name}_noisy.wav'), noisy, fs)
        sf.write(os.path.join(audio_dir, f'{base_name}_specsub_enhanced.wav'), enhanced, fs)

        # 绘制增强掩码图（用dB刻度，更直观）
        mask_filename = os.path.join(plot_dir, f'{base_name}_enhancement_mask.png')
        plt.figure(figsize=(10, 6))
        db_mask = librosa.amplitude_to_db(mag_enhanced, ref=np.max)
        img = plt.imshow(db_mask, aspect='auto', origin='lower', cmap='jet')
        plt.colorbar(img, format='%+2.0f dB')
        plt.title('SpecSub Enhancement Mask (dB)')
        plt.xlabel('Time (frames)')
        plt.ylabel('Frequency bins')
        plt.tight_layout()
        plt.savefig(mask_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Enhancement mask saved to: {mask_filename}')

        # 绘制频谱对比图
        plot_filename = os.path.join(plot_dir, f'{base_name}_spectrogram.png')
        noisy_energy, enhanced_energy, energy_reduction = plot_spectrogram_comparison(
            noisy, enhanced, plot_filename, fs=fs
        )

        # 绘制详细对比图
        detailed_filename = os.path.join(plot_dir, f'{base_name}_detailed.png')
        plot_detailed_comparison(noisy, enhanced, detailed_filename, fs=fs)

        # 记录结果
        results_summary.append({
            'filename': base_name,
            'noisy_energy': noisy_energy,
            'enhanced_energy': enhanced_energy,
            'energy_reduction': energy_reduction,
            'noisy_audio': os.path.join(audio_dir, f'{base_name}_noisy.wav'),
            'enhanced_audio': os.path.join(audio_dir, f'{base_name}_specsub_enhanced.wav'),
            'spectrogram': plot_filename,
            'detailed_plot': detailed_filename,
            'enhancement_mask': mask_filename
        })

        print(f"\nProcessed {base_name} successfully!")
        print(f"  Energy reduction: {energy_reduction:.2f} dB (higher = better noise suppression)")

    # 生成汇总报告
    if results_summary:
        summary_file = os.path.join(output_dir, 'evaluation_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("SpecSub Audio Enhancement Evaluation Summary (Optimized Version)\n")
            f.write("=" * 60 + "\n\n")
            for res in results_summary:
                f.write(f"File: {res['filename']}\n")
                f.write(f"  Noisy RMS Energy: {res['noisy_energy']:.4f}\n")
                f.write(f"  Enhanced RMS Energy: {res['enhanced_energy']:.4f}\n")
                f.write(f"  Energy Reduction: {res['energy_reduction']:.2f} dB\n")
                f.write(f"  Enhanced Audio: {res['enhanced_audio']}\n")
                f.write("-" * 60 + "\n")

            if len(results_summary) > 1:
                avg_noisy = np.mean([r['noisy_energy'] for r in results_summary])
                avg_enhanced = np.mean([r['enhanced_energy'] for r in results_summary])
                avg_reduction = np.mean([r['energy_reduction'] for r in results_summary])
                f.write(f"\nAverage Results:\n")
                f.write(f"  Avg Noisy Energy: {avg_noisy:.4f}\n")
                f.write(f"  Avg Enhanced Energy: {avg_enhanced:.4f}\n")
                f.write(f"  Avg Energy Reduction: {avg_reduction:.2f} dB\n")

        print(f"\nSummary report saved to: {summary_file}")
    print("\nAll processing completed! Check results in: ", output_dir)


if __name__ == "__main__":
    main()