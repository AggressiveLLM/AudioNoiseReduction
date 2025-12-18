import torch
import torch.nn as nn
import numpy as np
from model import Generator
from hparams import hparams
from dataset import emphasis
import glob
import soundfile as sf
import os
import librosa
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.linalg import norm


def enh_segan(model, noisy, para):
    win_len = para.win_len

    # 处理不足一个帧长的部分
    N_slice = len(noisy) // win_len
    if len(noisy) % win_len != 0:
        short = win_len - len(noisy) % win_len
        temp_noisy = np.pad(noisy, (0, short), 'wrap')
        N_slice = N_slice + 1
    else:
        temp_noisy = noisy.copy()

    slices = temp_noisy.reshape(N_slice, win_len)
    enh_slice = np.zeros(slices.shape)

    # 逐帧处理
    for n in range(N_slice):
        m_slice = slices[n]

        # 预加重
        m_slice = emphasis(m_slice)
        m_slice = np.expand_dims(m_slice, axis=(0, 1))
        m_slice = torch.from_numpy(m_slice)

        # 生成噪声向量
        z = nn.init.normal_(torch.Tensor(1, para.size_z[0], para.size_z[1]))

        # 生成增强语音
        model.eval()
        with torch.no_grad():
            generated_slice = model(m_slice, z)
        generated_slice = generated_slice.numpy()

        # 反预加重
        generated_slice = emphasis(generated_slice[0, 0, :], pre=False)
        enh_slice[n] = generated_slice

    # 合并所有帧
    enh_speech = enh_slice.reshape(N_slice * win_len)
    return enh_speech[:len(noisy)]


def get_snr(clean, noisy):
    noise = noisy - clean
    clean_norm = norm(clean)
    noise_norm = norm(noise)

    # 避免除以零
    if noise_norm == 0:
        return float('inf')

    snr = 20 * np.log10(clean_norm / (noise_norm + 1e-7))
    return snr


def calculate_rms_energy(signal):
    """计算信号的RMS能量"""
    return np.sqrt(np.mean(signal ** 2))


def plot_segan_detailed_comparison(clean, noisy, enh, output_filename, fs=16000, n_epoch=80):
    """绘制SEGAN的详细对比图 - 优化颜色映射版本"""
    # 确保所有音频长度一致
    min_len = min(len(clean), len(noisy), len(enh))
    clean = clean[:min_len]
    noisy = noisy[:min_len]
    enh = enh[:min_len]

    # 计算能量和SNR
    clean_energy = calculate_rms_energy(clean)
    noisy_energy = calculate_rms_energy(noisy)
    enh_energy = calculate_rms_energy(enh)

    snr_input = get_snr(clean, noisy)
    snr_output = get_snr(clean, enh)
    snr_improvement = snr_output - snr_input

    # 计算频谱数据以确定颜色映射范围
    def compute_spectral_data(signal, nfft=512, fs=16000):
        Pxx, freqs, bins, im = plt.specgram(signal, NFFT=nfft, Fs=fs, cmap='jet')
        plt.close()
        # 转换为dB
        Pxx_db = 10 * np.log10(Pxx + 1e-10)
        return Pxx_db

    # 计算动态范围
    Pxx_clean = compute_spectral_data(clean, 512, fs)
    Pxx_noisy = compute_spectral_data(noisy, 512, fs)
    Pxx_enh = compute_spectral_data(enh, 512, fs)

    # 确定合适的颜色范围
    all_data = np.concatenate([Pxx_clean.flatten(), Pxx_noisy.flatten(), Pxx_enh.flatten()])
    vmin = np.percentile(all_data, 5)  # 5%分位数
    vmax = np.percentile(all_data, 95)  # 95%分位数

    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'SEGAN Enhancement Analysis (Epoch {n_epoch})\n'
                 f'Input SNR: {snr_input:.2f} dB, Output SNR: {snr_output:.2f} dB, Improvement: {snr_improvement:.2f} dB',
                 fontsize=14, fontweight='bold', y=1.02)

    # 1. 频谱图对比 - 使用类似FRCRN的颜色映射
    plt.subplot(3, 3, 1)
    plt.specgram(clean, NFFT=512, Fs=fs, cmap='jet', vmin=vmin, vmax=vmax)
    plt.title('Clean Speech', fontsize=11, fontweight='bold')
    plt.ylabel('Frequency (Hz)', fontsize=9)
    plt.colorbar(format='%+2.0f dB', shrink=0.8)

    plt.subplot(3, 3, 2)
    plt.specgram(noisy, NFFT=512, Fs=fs, cmap='jet', vmin=vmin, vmax=vmax)
    plt.title(f'Noisy Speech (SNR: {snr_input:.2f} dB)', fontsize=11, fontweight='bold')
    plt.colorbar(format='%+2.0f dB', shrink=0.8)

    plt.subplot(3, 3, 3)
    plt.specgram(enh, NFFT=512, Fs=fs, cmap='jet', vmin=vmin, vmax=vmax)
    plt.title(f'SEGAN Enhanced (SNR: {snr_output:.2f} dB)', fontsize=11, fontweight='bold')
    plt.colorbar(format='%+2.0f dB', shrink=0.8)

    # 2. 波形对比图
    t = np.arange(min_len) / fs

    plt.subplot(3, 3, 4)
    plt.plot(t, clean, 'g-', alpha=0.8, linewidth=0.8)
    plt.title('Clean Waveform', fontsize=11)
    plt.xlabel('Time (s)', fontsize=9)
    plt.ylabel('Amplitude', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, t[-1]])

    plt.subplot(3, 3, 5)
    plt.plot(t, noisy, 'r-', alpha=0.8, linewidth=0.8)
    plt.title(f'Noisy Waveform (Energy: {noisy_energy:.4f})', fontsize=11)
    plt.xlabel('Time (s)', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, t[-1]])

    plt.subplot(3, 3, 6)
    plt.plot(t, enh, 'b-', alpha=0.8, linewidth=0.8)
    plt.title(f'Enhanced Waveform (Energy: {enh_energy:.4f})', fontsize=11)
    plt.xlabel('Time (s)', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, t[-1]])

    # 3. 频谱包络对比
    plt.subplot(3, 3, 7)
    freqs = np.fft.rfftfreq(min_len, 1 / fs)
    clean_fft = np.abs(np.fft.rfft(clean))
    noisy_fft = np.abs(np.fft.rfft(noisy))
    enh_fft = np.abs(np.fft.rfft(enh))

    plt.plot(freqs[:len(freqs) // 2], 20 * np.log10(clean_fft[:len(freqs) // 2] + 1e-7),
             'g-', alpha=0.7, label='Clean', linewidth=1.0)
    plt.plot(freqs[:len(freqs) // 2], 20 * np.log10(noisy_fft[:len(freqs) // 2] + 1e-7),
             'r-', alpha=0.7, label='Noisy', linewidth=1.0)
    plt.plot(freqs[:len(freqs) // 2], 20 * np.log10(enh_fft[:len(freqs) // 2] + 1e-7),
             'b-', alpha=0.8, label='Enhanced', linewidth=1.2)
    plt.title('Spectral Envelope (0-8kHz)', fontsize=11)
    plt.xlabel('Frequency (Hz)', fontsize=9)
    plt.ylabel('Magnitude (dB)', fontsize=9)
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.ylim([-60, 20])  # 设置合理的y轴范围

    # 4. 能量对比柱状图
    plt.subplot(3, 3, 8)
    energies = [clean_energy, noisy_energy, enh_energy]
    labels = ['Clean', 'Noisy', 'Enhanced']
    colors = ['green', 'red', 'blue']

    bars = plt.bar(labels, energies, color=colors, alpha=0.7)
    plt.title('RMS Energy Comparison', fontsize=11)
    plt.ylabel('RMS Energy', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')

    # 在柱状图上添加数值
    for bar, energy in zip(bars, energies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{energy:.4f}', ha='center', va='bottom', fontsize=8)

    # 5. SNR改进雷达图
    plt.subplot(3, 3, 9, polar=True)
    categories = ['Input SNR', 'Output SNR', 'Improvement']
    values = [snr_input, snr_output, snr_improvement]

    # 确保值在合理范围内
    values_normalized = np.array(values)
    values_normalized = (values_normalized - min(values_normalized)) / (
                max(values_normalized) - min(values_normalized) + 1e-7)

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    values_normalized = np.append(values_normalized, values_normalized[0])

    ax = plt.gca()
    ax.plot(angles, values_normalized, 'b-', linewidth=2)
    ax.fill(angles, values_normalized, 'b', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_title('SNR Performance', fontsize=11, y=1.1)
    ax.set_ylim([0, 1])  # 归一化雷达图

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Detailed comparison plot saved to: {output_filename}")
    print(f"Input SNR: {snr_input:.2f} dB, Output SNR: {snr_output:.2f} dB, Improvement: {snr_improvement:.2f} dB")
    print(f"Clean energy: {clean_energy:.4f}, Noisy energy: {noisy_energy:.4f}, Enhanced energy: {enh_energy:.4f}")

    return snr_input, snr_output, snr_improvement


def plot_segan_simple_comparison(clean, noisy, enh, output_filename, fs=16000, n_epoch=80):
    """绘制简化的SEGAN对比图 - 类似FRCRN风格"""
    min_len = min(len(clean), len(noisy), len(enh))
    clean = clean[:min_len]
    noisy = noisy[:min_len]
    enh = enh[:min_len]

    # 计算能量和SNR
    clean_energy = calculate_rms_energy(clean)
    noisy_energy = calculate_rms_energy(noisy)
    enh_energy = calculate_rms_energy(enh)

    snr_input = get_snr(clean, noisy)
    snr_output = get_snr(clean, enh)
    snr_improvement = snr_output - snr_input

    # 创建图形
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'SEGAN Enhancement Analysis (Epoch {n_epoch})', fontsize=14, fontweight='bold')

    # 1. 干净语音频谱
    plt.subplot(3, 1, 1)
    plt.specgram(clean, NFFT=512, Fs=fs, cmap='jet')
    plt.title(f'Clean Speech Spectrogram (Energy: {clean_energy:.4f})',
              fontsize=12, fontweight='bold')
    plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.colorbar(format='%+2.0f dB')

    # 2. 带噪语音频谱
    plt.subplot(3, 1, 2)
    plt.specgram(noisy, NFFT=512, Fs=fs, cmap='jet')
    plt.title(f'Noisy Speech Spectrogram (SNR: {snr_input:.2f} dB, Energy: {noisy_energy:.4f})',
              fontsize=12, fontweight='bold')
    plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.colorbar(format='%+2.0f dB')

    # 3. 增强语音频谱
    plt.subplot(3, 1, 3)
    plt.specgram(enh, NFFT=512, Fs=fs, cmap='jet')
    plt.title(f'SEGAN Enhanced Spectrogram (SNR: {snr_output:.2f} dB, Energy: {enh_energy:.4f})',
              fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'Simple spectrogram saved to: {output_filename}')
    print(f'Input SNR: {snr_input:.2f} dB, Output SNR: {snr_output:.2f} dB, Improvement: {snr_improvement:.2f} dB')

    return snr_input, snr_output, snr_improvement


# 模型加载函数
def load_model(model_file):
    """安全加载模型函数"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"Loading model from: {model_file}")

    # 检查文件是否存在
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    generator = Generator()


    try:

        state_dict = torch.load(model_file, map_location='cpu', weights_only=False)
        generator.load_state_dict(state_dict)
        print("Model loaded successfully with weights_only=False")
    except TypeError:

        try:
            state_dict = torch.load(model_file, map_location='cpu')
            generator.load_state_dict(state_dict)
            print("Model loaded successfully with default loading")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    return generator


if __name__ == "__main__":
    para = hparams()

    # 主输出目录
    main_output_dir = 'SEGAN_Evaluation_epoch80_cond0.0260'
    os.makedirs(main_output_dir, exist_ok=True)

    # 创建子目录
    summary_dir = os.path.join(main_output_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    # 加载模型
    n_epoch = 80
    model_file = "checkpoints_80/G_epoch80_cond0.0260.pkl"

    # 安全加载模型
    try:
        generator = load_model(model_file)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Creating a dummy model for testing...")
        generator = Generator()
        print("Warning: Using dummy model, enhancement quality may be poor")

    path_test_clean = '/root/autodl-tmp/Audio Noise Reduction/SEGAN/clean_testset_wav'
    path_test_noisy = '/root/autodl-tmp/Audio Noise Reduction/SEGAN/noisy_testset_wav'
    test_clean_wavs = glob.glob(path_test_clean + '/*wav')
    fs = para.fs

    # 结果汇总
    results_summary = []

    for clean_file in test_clean_wavs:
        name = os.path.split(clean_file)[-1]

        # 调试模式：只处理特定文件
        # if name != "p232_092.wav":
        #     continue

        noisy_file = os.path.join(path_test_noisy, name)
        if not os.path.isfile(noisy_file):
            continue

        # 读取音频
        clean, _ = librosa.load(clean_file, sr=fs, mono=True)
        noisy, _ = librosa.load(noisy_file, sr=fs, mono=True)

        # 确保长度一致
        min_len = min(len(clean), len(noisy))
        clean = clean[:min_len]
        noisy = noisy[:min_len]

        # 计算输入SNR
        snr_input = get_snr(clean, noisy)
        print(f"{name} - Input SNR: {snr_input:.2f} dB")

        # 只处理低SNR的音频
        if snr_input < 3.0:
            print(f'Processing {name} with SNR {snr_input:.2f} dB')

            # 获取增强语音
            enh = enh_segan(generator, noisy, para)

            # 为每个音频创建单独的文件夹
            base_name = name[:-4]  # 去掉.wav扩展名
            audio_dir = os.path.join(main_output_dir, base_name, 'audio')
            plot_dir = os.path.join(main_output_dir, base_name, 'plots')

            os.makedirs(audio_dir, exist_ok=True)
            os.makedirs(plot_dir, exist_ok=True)

            # 保存音频文件
            sf.write(os.path.join(audio_dir, 'noisy.wav'), noisy, fs)
            sf.write(os.path.join(audio_dir, 'clean.wav'), clean, fs)
            sf.write(os.path.join(audio_dir, 'enh.wav'), enh, fs)

            # 生成详细对比图
            try:
                plot_filename = os.path.join(plot_dir, f'{base_name}_detailed_analysis.png')
                snr_input, snr_output, snr_improvement = plot_segan_detailed_comparison(
                    clean, noisy, enh, plot_filename, fs, n_epoch=n_epoch
                )
            except Exception as e:
                print(f"Error generating detailed plot: {e}")
                # 尝试生成简化版
                try:
                    plot_filename = os.path.join(plot_dir, f'{base_name}_simple_comparison.png')
                    snr_input, snr_output, snr_improvement = plot_segan_simple_comparison(
                        clean, noisy, enh, plot_filename, fs, n_epoch=n_epoch
                    )
                except Exception as e2:
                    print(f"Error generating simple plot: {e2}")
                    continue

            # 记录结果
            result = {
                'filename': name,
                'base_name': base_name,
                'snr_input': snr_input,
                'snr_output': snr_output,
                'snr_improvement': snr_improvement,
                'audio_dir': audio_dir,
                'plot_dir': plot_dir,
                'clean_file': os.path.join(audio_dir, 'clean.wav'),
                'noisy_file': os.path.join(audio_dir, 'noisy.wav'),
                'enh_file': os.path.join(audio_dir, 'enh.wav'),
                'plot_file': plot_filename
            }
            results_summary.append(result)

            print(f"Completed processing for {name}")
            print("-" * 60)

    # 生成汇总报告
    if results_summary:
        print("\n" + "=" * 60)
        print("SEGAN ENHANCEMENT EVALUATION SUMMARY")
        print("=" * 60)

        summary_file = os.path.join(summary_dir, 'evaluation_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("SEGAN Speech Enhancement Evaluation Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: G_epoch{n_epoch}_cond0.0260.pkl\n")
            f.write(f"Sampling Rate: {fs} Hz\n")
            f.write(f"Total files processed: {len(results_summary)}\n\n")

            f.write("=" * 60 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("=" * 60 + "\n\n")

            for result in results_summary:
                f.write(f"File: {result['filename']}\n")
                f.write(f"  Base name: {result['base_name']}\n")
                f.write(f"  Input SNR: {result['snr_input']:.2f} dB\n")
                f.write(f"  Output SNR: {result['snr_output']:.2f} dB\n")
                f.write(f"  SNR Improvement: {result['snr_improvement']:.2f} dB\n")
                f.write(f"  Audio directory: {result['audio_dir']}\n")
                f.write(f"    - Clean: {result['clean_file']}\n")
                f.write(f"    - Noisy: {result['noisy_file']}\n")
                f.write(f"    - Enhanced: {result['enh_file']}\n")
                f.write(f"  Plot directory: {result['plot_dir']}\n")
                f.write(f"    - Analysis plot: {result['plot_file']}\n")
                f.write("-" * 60 + "\n")

            # 计算统计信息
            if len(results_summary) > 1:
                avg_snr_input = np.mean([r['snr_input'] for r in results_summary])
                avg_snr_output = np.mean([r['snr_output'] for r in results_summary])
                avg_snr_improvement = np.mean([r['snr_improvement'] for r in results_summary])

                f.write("\n" + "=" * 60 + "\n")
                f.write("STATISTICAL SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Average Input SNR: {avg_snr_input:.2f} dB\n")
                f.write(f"Average Output SNR: {avg_snr_output:.2f} dB\n")
                f.write(f"Average SNR Improvement: {avg_snr_improvement:.2f} dB\n")

                # 最佳和最差改进
                best_improvement = max(results_summary, key=lambda x: x['snr_improvement'])
                worst_improvement = min(results_summary, key=lambda x: x['snr_improvement'])

                f.write(f"\nBest Improvement: {best_improvement['filename']}\n")
                f.write(f"  Improvement: {best_improvement['snr_improvement']:.2f} dB\n")
                f.write(f"  From {best_improvement['snr_input']:.2f} dB to {best_improvement['snr_output']:.2f} dB\n")

                f.write(f"\nWorst Improvement: {worst_improvement['filename']}\n")
                f.write(f"  Improvement: {worst_improvement['snr_improvement']:.2f} dB\n")
                f.write(f"  From {worst_improvement['snr_input']:.2f} dB to {worst_improvement['snr_output']:.2f} dB\n")

        print(f"\nEvaluation complete!")
        print(f"All results are saved in: {main_output_dir}")
        print(f"Summary report: {summary_file}")

        # 打印简要统计信息
        print("\n" + "=" * 60)
        print("BRIEF STATISTICS")
        print("=" * 60)
        if len(results_summary) > 1:
            print(f"Average SNR Improvement: {avg_snr_improvement:.2f} dB")
    else:
        print("No audio files were processed.")