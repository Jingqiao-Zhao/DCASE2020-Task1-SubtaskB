# DCASE2020-Task1-SubtaskB

## Table of Contents

- [Background](#background)
- [Install](#install)
- [The proposed method](#the-proposed-method)
- [Results](#results)
- [References](#references)

## Background
The [DCASE2020](http://dcase.community/) dataset consists of 10 classes sounds captured in airport, shopping mall, metro station, pedestrian street, public,square, street traffic, tram, bus, metro and park. This challenge provides two datasets, development and evaluation, for algorithm development. The sub-task B of TAU Urban Acoustic Scenes 2020 dataset contains 40 hour audio recordings which are balanced between classes and recorded at 48kHz sampling rate with 24-bit resolution in stereo. Each sound recording was spitted into 10-second audio samples.

## Install
Download the [data](https://doi.org/10.5281/zenodo.3670185) and Change the path in config.py to your own.
```sh
$ pip install -r requirements.txt
```
```sh
$ python Data_generator.py
```
Then
```sh
$ python Train.py

```
## The proposed method
### Specaugment
Without sufficient training data, it is crucial to apply data augmentation to the existing training samples thus to improve the performance of a learning-based method by better exploiting the data in our hand. In sound recognition, traditional data augmentation methods include deformation of sound waves and background noise jetting. Different data augmentation methods are applied to each individual training sample. With data augmentation, we can train a network with better performance by synthesizing new training samples from the original ones. However, existing data augmentation methods, such as ASR, increase computational complexity and often require additional data.

![image1](https://github.com/Jingqiao-Zhao/DCASE2020-Task1-SubtaskB/blob/master/fig5_2.png)
To be more specific, we use SpecAugmen to modify spectrum maps by distorting time domain signal, masking the frequency domain channel and the time domain channel. This data augmentation method can be used to increase the robustness of the trained network to combat deformations on the time domain and partial fragment loss on the frequency domain. In this figure, we give an example of SpecAugment. 

![image2](https://github.com/Jingqiao-Zhao/DCASE2020-Task1-SubtaskB/blob/master/fig6_2.png)
### Depthwise convolution

![image3](https://github.com/Jingqiao-Zhao/DCASE2020-Task1-SubtaskB/blob/master/figure3.pdf)

## Results
## References
D. S. Park, W. Chan, Y. Zhang, C.-C. Chiu, B. Zoph, E. D.Cubuk, and Q. V. Le, “Specaugment: A simple data augmentation method for automatic speech recognition,” arXivpreprint arXiv:1904.08779, 2019.

Y. Tang, Y. Wang, Y. Xu, B. Shi, C. Xu, C. Xu, and C. Xu,“Beyond dropout: Feature map distortion to regularize deep neural networks,” arXiv preprint arXiv:2002.11022, 2020.

A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang,T. Weyand, M. Andreetto, and H. Adam, “Mobilenets: Efficient convolutional neural networks for mobile vision applications,” arXiv preprint arXiv:1704.04861, 2017.




