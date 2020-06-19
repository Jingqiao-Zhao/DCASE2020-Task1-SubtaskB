sample_rate = 32000
window_size = 1024
hop_size = 500  # So that there are 64 frames per second
mel_bins = 64
fmin = 50  # Hz
fmax = 14000  # Hz
batch_size=32
path='C:\\data\\TAU-urban-acoustic-scenes-2020-3class-development'
epochs = 100

frames_per_second = sample_rate // hop_size
audio_duration = 10  # Audio recordings in DCASE2019 Task1 are all 10 seconds
frames_num = frames_per_second * audio_duration
total_samples = sample_rate * audio_duration
labels = ['indoor', 'outdoor', 'transport']

classes_num = len(labels)





