import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import os
import json
import io
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.linalg import norm
from ans.frcrn import FRCRN


def model_load(file_model, net):
    """FRCRN模型加载函数"""
    state_dict = net.state_dict()
    save_state_dic = torch.load(file_model, map_location='cpu')

    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = save_state_dic[k]
        except:
            print(k, 'not find')
            new_state_dict[k] = v

    net.load_state_dict(new_state_dict)
    return net


def audio_norm(x):
    """音频归一化"""
    rms = (x ** 2).mean() ** 0.5
    scalar = 10 ** (-25 / 20) / rms
    x = x * scalar
    pow_x = x ** 2
    avg_pow_x = pow_x.mean()
    rmsx = pow_x[pow_x > avg_pow_x].mean() ** 0.5
    scalarx = 10 ** (-25 / 20) / rmsx
    x = x * scalarx
    return x


def audioread(file_wav):
    """音频读取函数"""
    SAMPLE_RATE = 16000
    with open(file_wav, 'rb') as f:
        content = f.read()
    data1, fs = sf.read(io.BytesIO(content))
    if len(data1.shape) > 1:
        data1 = data1[:, 0]
    if fs != 16000:
        data1 = librosa.resample(data1, orig_sr=fs, target_sr=SAMPLE_RATE)
        fs = SAMPLE_RATE
    data1 = audio_norm(data1)
    data = data1.astype(np.float32)
    inputs = np.reshape(data, [1, data.shape[0]])
    return inputs, fs


def simple_audioread(file_wav, target_sr=16000):
    """简化的音频读取函数"""
    try:
        data, fs = sf.read(file_wav)
        if len(data.shape) > 1:
            data = data[:, 0]  # 取第一个声道
        if fs != target_sr:
            data = librosa.resample(data, orig_sr=fs, target_sr=target_sr)
            fs = target_sr
        return data, fs
    except Exception as e:
        print(f"Error reading {file_wav}: {e}")
        return None, None


def do_enh(data, model, device):
    """FRCRN增强函数"""
    ndarray = data
    if isinstance(ndarray, torch.Tensor):
        ndarray = ndarray.cpu().numpy()
    nsamples = data.shape[1]
    decode_do_segement = False
    window = 16000
    stride = int(window * 0.75)
    print(f'inputs: {ndarray.shape}')

    b, t = ndarray.shape
    if t > window * 120:
        decode_do_segement = True

    # 补零处理
    if t < window:
        ndarray = np.concatenate([ndarray, np.zeros((ndarray.shape[0], window - t))], 1)
    elif t < window + stride:
        padding = window + stride - t
        print(f'padding: {padding}')
        ndarray = np.concatenate([ndarray, np.zeros((ndarray.shape[0], padding))], 1)
    else:
        if (t - window) % stride != 0:
            padding = t - (t - window) // stride * stride
            print(f'padding: {padding}')
            ndarray = np.concatenate([ndarray, np.zeros((ndarray.shape[0], padding))], 1)

    print(f'inputs after padding: {ndarray.shape}')

    with torch.no_grad():
        ndarray = torch.from_numpy(np.float32(ndarray)).to(device)
        b, t = ndarray.shape

        if decode_do_segement:
            outputs = np.zeros(t)
            give_up_length = (window - stride) // 2
            current_idx = 0
            while current_idx + window <= t:
                print(f'current_idx: {current_idx}')
                tmp_input = dict(noisy=ndarray[:, current_idx:current_idx + window])
                tmp_output = model(tmp_input, )[4][0].cpu().numpy()
                end_index = current_idx + window - give_up_length
                if current_idx == 0:
                    outputs[current_idx:end_index] = tmp_output[:-give_up_length]
                else:
                    outputs[current_idx + give_up_length:end_index] = tmp_output[give_up_length:-give_up_length]
                current_idx += stride
        else:
            outputs = model.infer(ndarray)[0].cpu().numpy()

    outputs = (outputs[:nsamples] * 32768).astype(np.int16)
    return outputs


def calculate_rms_energy(signal):
    """计算信号的RMS能量"""
    return np.sqrt(np.mean(signal ** 2))


def plot_frcrn_spectrogram_comparison(noisy, frcrn_enh, output_filename, fs=16000):
    """绘制FRCRN的频谱对比图（仅噪声输入）"""
    # 确保所有音频长度一致
    min_len = min(len(noisy), len(frcrn_enh))
    noisy = noisy[:min_len]
    frcrn_enh = frcrn_enh[:min_len]

    # 计算能量变化
    noisy_energy = calculate_rms_energy(noisy)
    enhanced_energy = calculate_rms_energy(frcrn_enh)
    energy_reduction = 10 * np.log10((noisy_energy ** 2) / (enhanced_energy ** 2 + 1e-7))

    # 创建图形
    fig = plt.figure(figsize=(15, 10))

    # 1. 原始带噪语音频谱
    plt.subplot(3, 1, 1)
    plt.specgram(noisy, NFFT=512, Fs=fs, cmap='jet')
    plt.title(f'Noisy Speech Spectrogram (Energy: {noisy_energy:.4f})',
              fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(format='%+2.0f dB')

    # 2. FRCRN增强语音频谱
    plt.subplot(3, 1, 2)
    plt.specgram(frcrn_enh, NFFT=512, Fs=fs, cmap='jet')
    plt.title(f'FRCRN Enhanced Spectrogram (Energy: {enhanced_energy:.4f}, Reduction: {energy_reduction:.2f} dB)',
              fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(format='%+2.0f dB')

    # 3. 波形对比图
    plt.subplot(3, 1, 3)
    t = np.arange(min_len) / fs

    # 归一化波形以便对比
    noisy_norm = noisy / np.max(np.abs(noisy) + 1e-7)
    frcrn_norm = frcrn_enh / np.max(np.abs(frcrn_enh) + 1e-7)

    plt.plot(t, noisy_norm, 'r-', alpha=0.6, label='Noisy', linewidth=0.8)
    plt.plot(t, frcrn_norm, 'b-', alpha=0.8, label='FRCRN Enhanced', linewidth=1.2)

    plt.title('Waveform Comparison (Normalized)', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Normalized Amplitude', fontsize=10)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, t[-1]])
    plt.ylim([-1.2, 1.2])

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'FRCRN spectrogram saved to: {output_filename}')
    print(f'Noisy energy: {noisy_energy:.4f}')
    print(f'Enhanced energy: {enhanced_energy:.4f}')
    print(f'Energy reduction: {energy_reduction:.2f} dB')

    return noisy_energy, enhanced_energy, energy_reduction


def plot_frcrn_detailed_comparison(noisy, frcrn_enh, output_filename, fs=16000):
    """绘制更详细的FRCRN对比图"""
    min_len = min(len(noisy), len(frcrn_enh))
    noisy = noisy[:min_len]
    frcrn_enh = frcrn_enh[:min_len]

    # 创建更详细的图形
    fig = plt.figure(figsize=(16, 12))

    # 1. 原始带噪语音频谱
    plt.subplot(3, 2, 1)
    plt.specgram(noisy, NFFT=512, Fs=fs, cmap='jet')
    plt.title('Noisy Speech Spectrogram', fontsize=11)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # 2. FRCRN增强语音频谱
    plt.subplot(3, 2, 2)
    plt.specgram(frcrn_enh, NFFT=512, Fs=fs, cmap='jet')
    plt.title('FRCRN Enhanced Spectrogram', fontsize=11)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # 3. 波形对比图
    plt.subplot(3, 2, 3)
    t = np.arange(min_len) / fs
    plt.plot(t, noisy, 'r-', alpha=0.7, label='Noisy', linewidth=0.8)
    plt.title('Noisy Waveform', fontsize=11)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 4)
    plt.plot(t, frcrn_enh, 'b-', alpha=0.8, label='Enhanced', linewidth=1.0)
    plt.title('FRCRN Enhanced Waveform', fontsize=11)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    # 4. 频谱包络对比
    plt.subplot(3, 2, 5)
    freqs = np.fft.rfftfreq(min_len, 1 / fs)
    noisy_fft = np.abs(np.fft.rfft(noisy))
    enhanced_fft = np.abs(np.fft.rfft(frcrn_enh))

    plt.plot(freqs[:len(freqs) // 4], 20 * np.log10(noisy_fft[:len(freqs) // 4] + 1e-7),
             'r-', alpha=0.7, label='Noisy', linewidth=1.0)
    plt.plot(freqs[:len(freqs) // 4], 20 * np.log10(enhanced_fft[:len(freqs) // 4] + 1e-7),
             'b-', alpha=0.8, label='Enhanced', linewidth=1.2)
    plt.title('Spectral Envelope (0-4kHz)', fontsize=11)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)

    # 5. 时频能量分布
    plt.subplot(3, 2, 6)
    noisy_energy_per_band = []
    enhanced_energy_per_band = []

    # 计算不同频带能量
    bands = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
    band_labels = ['0-1k', '1k-2k', '2k-4k', '4k-8k']

    for i, (low, high) in enumerate(bands):
        low_idx = int(low * min_len / fs)
        high_idx = int(high * min_len / fs)

        noisy_band_energy = np.sum(noisy_fft[low_idx:high_idx] ** 2)
        enhanced_band_energy = np.sum(enhanced_fft[low_idx:high_idx] ** 2)

        noisy_energy_per_band.append(noisy_band_energy)
        enhanced_energy_per_band.append(enhanced_band_energy)

    x_pos = np.arange(len(bands))
    width = 0.35

    plt.bar(x_pos - width / 2, noisy_energy_per_band, width,
            label='Noisy', alpha=0.7, color='red')
    plt.bar(x_pos + width / 2, enhanced_energy_per_band, width,
            label='Enhanced', alpha=0.8, color='blue')

    plt.title('Frequency Band Energy Distribution', fontsize=11)
    plt.xlabel('Frequency Bands')
    plt.ylabel('Energy')
    plt.xticks(x_pos, band_labels)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'Detailed FRCRN comparison saved to: {output_filename}')


def main():
    # 创建输出目录
    output_dir = 'frcrn_Evaluation'
    audio_dir = os.path.join(output_dir, 'audio')
    plot_dir = os.path.join(output_dir, 'plots')

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # 加载FRCRN模型
    print("Loading FRCRN model...")
    file_model = os.path.join('pre_trained_model', 'pytorch_model.bin')
    file_json = os.path.join('pre_trained_model', 'configuration.json')

    with open(file_json) as f:
        data = json.load(f)
    hps = data['model']
    print(f"Model hyperparameters: {hps}")

    m_model = FRCRN(
        hps['complex'],
        hps['model_complexity'],
        hps['model_depth'],
        hps['log_amp'],
        hps['padding_mode'],
        hps['win_len'],
        hps['win_inc'],
        hps['fft_len'],
        hps['win_type']
    )
    m_model = model_load(file_model, m_model)

    # 设置测试文件
    test_noisy_files = [
        '/root/autodl-tmp/Audio Noise Reduction/FRCRN/video/speech_with_noise1.wav',

    ]

    # 结果汇总
    results_summary = []

    for noisy_file in test_noisy_files:
        if not os.path.exists(noisy_file):
            print(f"File not found: {noisy_file}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing: {os.path.basename(noisy_file)}")
        print(f"{'=' * 60}")

        # 读取噪声文件
        print("Loading noisy audio file...")
        noisy_audio, fs = simple_audioread(noisy_file, target_sr=16000)
        if noisy_audio is None:
            continue

        # 准备输入数据
        noisy_input = np.reshape(noisy_audio, [1, len(noisy_audio)])

        # 运行FRCRN增强
        print("Running FRCRN enhancement...")
        enhanced_audio = do_enh(noisy_input, m_model, 'cpu')
        enhanced_audio_float = enhanced_audio.astype(np.float32) / 32768.0

        # 保存音频文件
        print("Saving audio files...")
        base_name = os.path.splitext(os.path.basename(noisy_file))[0]

        sf.write(os.path.join(audio_dir, f'{base_name}_noisy.wav'),
                 noisy_audio, 16000)
        sf.write(os.path.join(audio_dir, f'{base_name}_frcrn_enhanced.wav'),
                 enhanced_audio_float, 16000)

        # 绘制频谱对比图
        print("Generating spectrogram comparison...")
        plot_filename = os.path.join(plot_dir, f'{base_name}_spectrogram.png')

        noisy_energy, enhanced_energy, energy_reduction = plot_frcrn_spectrogram_comparison(
            noisy_audio,
            enhanced_audio_float,
            plot_filename,
            fs=16000
        )

        # 绘制详细对比图
        print("Generating detailed comparison...")
        detailed_filename = os.path.join(plot_dir, f'{base_name}_detailed.png')
        plot_frcrn_detailed_comparison(
            noisy_audio,
            enhanced_audio_float,
            detailed_filename,
            fs=16000
        )

        # 记录结果
        result = {
            'filename': base_name,
            'noisy_energy': noisy_energy,
            'enhanced_energy': enhanced_energy,
            'energy_reduction': energy_reduction,
            'noisy_audio': os.path.join(audio_dir, f'{base_name}_noisy.wav'),
            'enhanced_audio': os.path.join(audio_dir, f'{base_name}_frcrn_enhanced.wav'),
            'spectrogram': plot_filename,
            'detailed_plot': detailed_filename
        }
        results_summary.append(result)

        print(f"Processed: {base_name}")
        print(f"  Noisy energy: {noisy_energy:.4f}")
        print(f"  Enhanced energy: {enhanced_energy:.4f}")
        print(f"  Energy reduction: {energy_reduction:.2f} dB")

    # 生成汇总报告
    if results_summary:
        print(f"\n{'=' * 60}")
        print("SUMMARY REPORT")
        print(f"{'=' * 60}")

        summary_file = os.path.join(output_dir, 'evaluation_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("FRCRN Audio Enhancement Evaluation Summary (Noisy Input Only)\n")
            f.write("=" * 60 + "\n\n")

            for result in results_summary:
                f.write(f"File: {result['filename']}\n")
                f.write(f"  Noisy energy: {result['noisy_energy']:.4f}\n")
                f.write(f"  Enhanced energy: {result['enhanced_energy']:.4f}\n")
                f.write(f"  Energy reduction: {result['energy_reduction']:.2f} dB\n")
                f.write(f"  Noisy audio: {result['noisy_audio']}\n")
                f.write(f"  Enhanced audio: {result['enhanced_audio']}\n")
                f.write(f"  Spectrogram: {result['spectrogram']}\n")
                f.write(f"  Detailed plot: {result['detailed_plot']}\n")
                f.write("-" * 60 + "\n")

            # 计算平均值
            if len(results_summary) > 1:
                avg_noisy_energy = np.mean([r['noisy_energy'] for r in results_summary])
                avg_enhanced_energy = np.mean([r['enhanced_energy'] for r in results_summary])
                avg_energy_reduction = np.mean([r['energy_reduction'] for r in results_summary])

                f.write(f"\nAverage Results:\n")
                f.write(f"  Average noisy energy: {avg_noisy_energy:.4f}\n")
                f.write(f"  Average enhanced energy: {avg_enhanced_energy:.4f}\n")
                f.write(f"  Average energy reduction: {avg_energy_reduction:.2f} dB\n")

        print(f"\nEvaluation complete! Results saved in: {output_dir}")
        print(f"Summary report: {summary_file}")

    print("\nFRCRN evaluation completed successfully!")


if __name__ == "__main__":
    main()