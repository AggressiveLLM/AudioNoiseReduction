import numpy as np
import librosa
import os


def wav_split(wav, win_length, strid):
    slices = []
    if len(wav) > win_length:
        for idx_end in range(win_length, len(wav), strid):
            idx_start = idx_end - win_length
            slice_wav = wav[idx_start:idx_end]
            slices.append(slice_wav)
        # 拼接最后一帧
        slices.append(wav[-win_length:])
    return slices


def save_slices(slices, name):
    name_list = []
    if len(slices) > 0:
        for i, slice_wav in enumerate(slices):
            name_slice = name + "_" + str(i) + '.npy'
            np.save(name_slice, slice_wav)
            name_list.append(name_slice)
    return name_list


if __name__ == "__main__":
    clean_wav_path = "/root/autodl-tmp/Audio Noise Reduction/SEGAN/clean_trainset_28spk_wav/"
    noisy_wav_path = "/root/autodl-tmp/Audio Noise Reduction/SEGAN/noisy_trainset_28spk_wav/"

    catch_train_clean = '/root/autodl-tmp/Audio Noise Reduction/SEGAN/catch_segan/clean'
    catch_train_noisy = '/root/autodl-tmp/Audio Noise Reduction/SEGAN/catch_segan/noisy'

    # 确保目录存在
    os.makedirs(catch_train_clean, exist_ok=True)
    os.makedirs(catch_train_noisy, exist_ok=True)


    script_dir = os.path.dirname(os.path.abspath(__file__))
    scp_dir = os.path.join(script_dir, "scp")
    os.makedirs(scp_dir, exist_ok=True)
    scp_file = os.path.join(scp_dir, "train_segan.scp")

    win_length = 16384
    strid = int(win_length / 2)

    # 处理统计
    processed_count = 0
    error_count = 0

    with open(scp_file, 'wt') as f:
        for filename in os.listdir(clean_wav_path):
            if not filename.endswith(".wav"):
                continue

            file_clean_name = os.path.join(clean_wav_path, filename)
            file_noisy_name = os.path.join(noisy_wav_path, filename)

            print("processing file %s" % (file_clean_name))

            if not os.path.exists(file_noisy_name):
                print("can not find file %s" % (file_noisy_name))
                error_count += 1
                continue

            try:
                clean_data, sr = librosa.load(file_clean_name, sr=16000, mono=True)
                noisy_data, sr = librosa.load(file_noisy_name, sr=16000, mono=True)

                if not len(clean_data) == len(noisy_data):
                    print("file length are not equal")
                    error_count += 1
                    continue


                base_name = os.path.splitext(filename)[0]

                # 干净语音分段+保存
                clean_slices = wav_split(clean_data, win_length, strid)
                clean_namelist = save_slices(clean_slices,
                                             os.path.join(catch_train_clean, base_name))

                # 噪声语音分段+保存
                noisy_slices = wav_split(noisy_data, win_length, strid)
                noisy_namelist = save_slices(noisy_slices,
                                             os.path.join(catch_train_noisy, base_name))

                # 确保两个列表长度相同
                min_length = min(len(clean_namelist), len(noisy_namelist))
                for i in range(min_length):
                    f.write("%s %s\n" % (clean_namelist[i], noisy_namelist[i]))

                processed_count += 1
                print(f"  Generated {min_length} slices")

            except Exception as e:
                print(f"  Error: {str(e)}")
                error_count += 1
                continue

    print(f"\n=== Processing Summary ===")
    print(f"Successfully processed: {processed_count} files")
    print(f"Failed files: {error_count}")
    print(f"SCP file saved to: {scp_file}")

    # 验证输出
    if os.path.exists(catch_train_clean):
        npy_files = [f for f in os.listdir(catch_train_clean) if f.endswith('.npy')]
        print(f"Clean slices generated: {len(npy_files)}")

    if os.path.exists(scp_file):
        with open(scp_file, 'r') as f:
            lines = f.readlines()
        print(f"SCP file lines: {len(lines)}")