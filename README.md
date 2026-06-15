# Intelligent Lung Sound Classification System

## Overview

This repository contains a reconstructed version of my undergraduate major project focused on automated lung sound classification for primary healthcare applications.

The system acquires respiratory signals using an Arduino-based data acquisition setup, performs signal preprocessing and spectrogram generation in Python, and classifies lung sounds using deep learning models including **VGG16** and **ResNet18**.

The original project integrated embedded hardware, digital signal processing, and machine learning to assist in distinguishing between different respiratory conditions.

## Features

* Real-time acquisition of lung sound data using Arduino UNO
* Serial communication between Arduino and Python
* Signal preprocessing and noise reduction
* Empirical Mode Decomposition (EMD)-based signal analysis
* Spectrogram generation for feature extraction
* Lung sound classification using VGG16
* Lung sound classification using ResNet18
* Classification into:

  * Healthy
  * Crackles
  * Wheezes
  * Crackles and Wheezes

## Technologies Used

* Python
* C++
* Arduino UNO
* NumPy
* SciPy
* PySerial
* Matplotlib
* PyWavelets
* PyEMD
* TensorFlow / Keras
* PyTorch

## Repository Structure

```
arduino/
    lung_sensor.ino

python/
    main.py

README.md
```

## Note

This repository is a reconstructed version of an academic project. The original source files were unavailable, so the implementation has been recreated from archived copies and project documentation.






