import pandas as pd
import numpy as np
import librosa
import os
import time
import sys
import config
from utilities import spec_augment_pytorch
import matplotlib.pyplot as plt
import pickle
import torch

def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]

def get_csv(csv_path):
    data = pd.read_csv(csv_path,sep='\t')
    data_dict={}
    for i in range(len(data['filename'])):
        data_dict[data['filename'][i]]=data['scene_label'][i]
    return data_dict


def read_audio(audio_path, target_fs=None):
    (audio, fs) = librosa.load(audio_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    return audio, fs


def calculate_feature_for_all_audio_files(csv_path,file_name):
    sample_rate = config.sample_rate
    window_size = config.window_size
    hop_size = config.hop_size
    mel_bins = config.mel_bins
    fmin = config.fmin
    fmax = config.fmax
    frames_per_second = config.frames_per_second
    frames_num = config.frames_num
    total_samples = config.total_samples
    path = config.path

    # Read metadata
    csv_dict = get_csv(csv_path)
    i = 0
    n = len(csv_dict.keys())
    print('Find %d Audio in Csv_File' % n)
    # creat feature_dict
    feature_data = np.ndarray([n, frames_num, mel_bins])
    feature_dict = {}
    # Extract features and targets 提取相关特征以及目标

    for key, value in csv_dict.items():
        audio_path = os.path.join(path, key)

        # Read audio
        (audio, _) = read_audio(
            audio_path=audio_path,
            target_fs=sample_rate)

        # Pad or truncate audio recording to the same length
        audio = pad_truncate_sequence(audio, total_samples)

        # Extract feature
        # feature = feature_extractor.transform(audio)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                         sr=sample_rate,
                                                         n_fft=2048,
                                                         n_mels=mel_bins,
                                                         win_length=window_size,
                                                         hop_length=hop_size,
                                                         fmax=fmax)

        shape = mel_spectrogram.shape

        mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1]))

        mel_spectrogram = torch.from_numpy(mel_spectrogram)
        warped_masked_spectrogram = spec_augment_pytorch.spec_augment(mel_spectrogram=mel_spectrogram)
        warped_masked_spectrogram_numpy = warped_masked_spectrogram[0].numpy().transpose()
        # Remove the extra log mel spectrogram frames caused by padding zero
        feature = warped_masked_spectrogram_numpy
        feature = feature[0: frames_num]

        feature_data[i] = feature
        i += 1
        print("\r", end="")
        print("Feature extraction progress: {}%: ".format(round(float(i * 100 / n), 3)), "▋" * int((i // (n / 20))),
              end="")
        sys.stdout.flush()
        time.sleep(0.5)

    print('\n----------------------------------------------------------------------------')
    print('Feature extraction completed.', '\nStart building a feature dictionary.')

    feature_dict['audio_name'] = list(csv_dict.keys())
    feature_dict['label'] = list(csv_dict.values())
    feature_dict['feature'] = feature_data
    print('Feature dictionary establishment complete.')
    output = open('{}_feature_dict.pkl'.format(file_name), 'wb')
    pickle.dump(feature_dict, output)
    print('Feature_dict has been save as {}_feature_dict.pkl'.format(file_name))

    return feature_dict

train = calculate_feature_for_all_audio_files(os.path.join(config.path,'evaluation_setup\\fold1_train.csv'),file_name='Train')
test = calculate_feature_for_all_audio_files(os.path.join(config.path,'evaluation_setup\\fold1_evaluate.csv'),file_name='Test')