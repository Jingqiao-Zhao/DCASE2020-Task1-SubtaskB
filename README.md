# DCASE2020-Task1-SubtaskB

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
	- [Generator](#generator)
- [Badge](#badge)
- [Example Readmes](#example-readmes)
- [Related Efforts](#related-efforts)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

##Background
The [DCASE2020](http://dcase.community/) dataset consists of 10 classes sounds captured in airport, shopping mall, metro station, pedestrian street, public,square, street traffic, tram, bus, metro and park. This challenge provides two datasets, development and evaluation, for algorithm development. The sub-task B of TAU Urban Acoustic Scenes 2020 dataset contains 40 hour audio recordings which are balanced between classes and recorded at 48kHz sampling rate with 24-bit resolution in stereo. Each sound recording was spitted into 10-second audio samples.

##Install

```sh
$ pip install -r requirements.txt
```
Change the path in config.py to your own.
```sh
$ python Data_generator.py
```
Then
```sh
$ python train.py

```
