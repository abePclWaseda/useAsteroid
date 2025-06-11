import librosa
import numpy as np
import matplotlib.pyplot as plt

file_name = "15f97caafef4a4b352e54781236cf2b4.wav"

y, sr = librosa.load(file_name)

from asteroid.models import ConvTasNet
import torch
import torchaudio

model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")

# 音声データの整形
waveform_torch = torch.tensor(y).unsqueeze(0)  # (1, 1, T)
resampler = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)
waveform_torch = resampler(waveform_torch)

# モデルで話者分離（出力：話者数 x サンプル数）
separated_sources = model.separate(waveform_torch)
separated_sources_np = separated_sources.cpu().numpy()

import soundfile as sf

# separated_sources_np: shape = (1, 2, T) → (2, T)
stereo_data = separated_sources_np[0]  # shape: (2, T)

# 転置して shape を (T, 2) にする → ステレオ形式
stereo_data = stereo_data.T  # shape: (T, 2)

# 保存
sf.write("separated_stereo.wav", stereo_data, samplerate=16000)