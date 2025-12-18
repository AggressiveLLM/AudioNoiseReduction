# AudioNoiseReduction

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/AggressiveLLM/AudioNoiseReduction.svg)](https://github.com/AggressiveLLM/AudioNoiseReduction/issues)

## 项目概述

`AudioNoiseReduction` 是一个旨在解决音频信号噪声干扰问题的先进库。该项目集成了四种不同的降噪模型，涵盖从传统信号处理到最前沿的深度学习技术。它能够广泛应用于语音识别、通信增强、音频修复等场景。

### 核心功能
- **多模型支持**：提供 FRCRN、SEGAN、SpecSub 和 Wiener 四种模型，满足从实际场景优化到实时处理的不同需求。
- **模块化设计**：支持自定义训练及模型参数调整以优化应用效果。
- **效果显著**：全面评估并对比模型的降噪性能（如 SNR 和 RMS 能量变化）。
- **易用性高**：代码清晰易读，简单快捷地集成到项目中。

---

## 目录
1. [实验环境](#实验环境)
2. [模型使用](#模型使用)
3. [效果展示](#效果展示)
4. [贡献方式](#贡献方式)
5. [许可证](#许可证)

---

## 实验环境

以下是本项目的运行配置和依赖，您可以根据当前环境安装对应版本，以便复现实验或进一步改进。

| 分类        | 配置                       |
|-------------|----------------------------|
| **Python**  | 3.12 (推荐 Ubuntu 22.04)  |
| **PyTorch** | 2.3.0                     |
| **CUDA**    | 12.1                      |
| **GPU**     | RTX 4090 (24GB)           |
| **CPU**     | 16 vCPU Intel Xeon Gold 6430 |
| **OS**      | Ubuntu Server 22.04       |

确保相关驱动及依赖已正确安装后，即可运行项目代码。推荐您使用支持 GPU 的设备以加速深度学习模型的推理和处理。

---

## 模型使用

`AudioNoiseReduction` 提供四种降噪模型，以下是每种模型的使用方法。

### 1. **FRCRN 模型**
频域递归神经网络（FRCRN），通过复杂频谱操作及深度网络实现高效语音增强。  
主要文件：`FRCRN/test.py`

```python
from frcrn.main import FRCRN

# 初始化模型与参数文件
model = FRCRN(model_file="path/to/pre_trained_model.bin")

# 处理带噪声文件
output_audio = model.enhance("noisy_audio.wav")
print("FRCRN 增强完成，输出文件位于：output_audio.wav")
```

### 2. **SEGAN 模型**
基于生成对抗网络（GAN）的语音增强模型。  
主要文件：`SEGAN/eval.py`

```python
from segan.eval import enh_segan

# 加载预训练模型
model = Generator.load_weights("path/to/segan_weights.bin")

# 执行语音增强
enhanced_audio = enh_segan(model, "path/to/noisy_input.wav")
print("SEGAN 增强完成，输出文件为：enhanced_audio.wav")
```

### 3. **SpecSub 模型**
经典频谱减法降噪算法，通过简单频谱运算实现快速降噪。  
主要文件：`SPECSUB/spessub_test.py`

```python
from specsub.spessub_test import specsub_enhancement

# 加载输入音频并执行增强
enhanced, _ = specsub_enhancement(noisy_input="noisy.wav", fs=16000)
print("SpecSub 增强完成，文件已保存。")
```

### 4. **Wiener 滤波模型**
基于信号统计特性的传统降噪方案。  
主要文件：`WIENER/test_wiener_1.py`

```python
from wiener.main import wiener_filter

# 配置参数并执行 Wiener 滤波增强
output_clean = wiener_filter(noisy_audio, clean_reference, params)
print("Wiener 滤波降噪完成。")
```

---

## 效果展示

以下为 **不同模型的降噪效果** 对比。模型增强前的音频包含明显的背景噪音，而增强后的音频清晰度显著提高。

### 1. 对比示例：FRCRN
**增强前 vs. 增强后**
![FRCRN Spectrogram](https://github.com/AggressiveLLM/blob/master/AudioNoiseReduction/FRCRN/frcrn_Evaluation/plots/speech_with_noise1_detailed.png)   

---

### 2. 对比示例：SEGAN
**增强前 vs. 增强后**
![SEGAN Spectrogram](AudioNoiseReduction\SEGAN\Result\p232_092\plots\p232_092_detailed_analysis.png)

---

### 3. 对比示例：SpecSub
**增强前 vs. 增强后**
![SpecSub Spectrogram](AudioNoiseReduction\SPECSUB\specsub_evaluation\plots\noisy_detailed.png)

---

### 4. 对比示例：Wiener
**增强前 vs. 增强后**
![Wiener Spectrogram](AudioNoiseReduction\WIENER\Result\wiener_filter_analysis_1.png)


---

## 贡献方式

我们欢迎社区贡献！贡献方式如下：
1. Fork 本项目代码。
2. 创建功能分支：`git checkout -b feature_branch`
3. 提交代码更改：`git commit -m '描述改动内容'`
4. 推送分支并创建 Pull Request。
5. 等待审查后合并至主分支。

详细贡献流程请参考 [Contributing Guide](CONTRIBUTING.md)。

---

## 许可证

`AudioNoiseReduction` 开源项目基于 [MIT License](LICENSE)。

---

让我们减少背景噪声的干扰，专注于清晰的声音，为语音处理领域创造更佳表现！
