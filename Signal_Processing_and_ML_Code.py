import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
import codecs
import serial
import time
import os
import librosa
import wave as wav
import pywt
from PIL import Image

from PyEMD import EMD
from scipy.signal import hilbert

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import load_model

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


# =========================
# SERIAL DATA ACQUISITION
# =========================

ser = serial.Serial('COM5', 9600, timeout=1)
time.sleep(2)

co = []

for i in range(2000):
    line = ser.readline()
    try:
        s = codecs.decode(line)
        content = s.strip()
        print(content)
        co.append(float(content))
    except:
        pass

ser.close()

num = [float(item) for item in co if item != '']
signal1 = np.array(num[1000:2000])


# =========================
# WAVELET DENOISING
# =========================

fs, data = wav.read("sample.wav")
data = data[::6]

if len(data.shape) > 1:
    data = data[:, 0]

wavelet = 'db4'
level = 4

coeffs = pywt.wavedec(data, wavelet, level=level)
reconstructed_signal = pywt.waverec(coeffs, wavelet)

reconstructed_signal2 = reconstructed_signal[:1000]
sample_entropy = statistics.mean(reconstructed_signal2)
print("Sample entropy:", sample_entropy)


# =========================
# EMD PROCESSING
# =========================

emd = EMD()
imfs = emd(signal1)

envelopes = []

for i in range(len(imfs)):
    analytic_signal = hilbert(imfs[i])
    amplitude_envelope = np.abs(analytic_signal)
    envelopes.append(amplitude_envelope)

heart_envelope = envelopes[2] if len(envelopes) > 2 else envelopes[0]
lung_signal = signal1 - heart_envelope


# =========================
# SPECTROGRAM GENERATION
# =========================

plt.figure()

plt.specgram(lung_signal, Fs=1764, cmap='inferno')
plt.axis('off')

plt.savefig('temp.png', bbox_inches='tight', pad_inches=0)
plt.close()

image = Image.open('temp.png')
image_resized = image.resize((224, 224))


# =========================
# SAVE IMAGE
# =========================

output_dir = "output_spectrograms"
os.makedirs(output_dir, exist_ok=True)

filename = input("Enter filename: ")

output_filename = os.path.join(output_dir, filename + ".png")
image_resized.save(output_filename)

os.remove('temp.png')


# =========================
# VGG16 INFERENCE
# =========================

model = load_model('lung_sound_classification_model.h5')

labels = ['crackles', 'wheezes', 'healthy', 'both']

img = Image.open(output_filename).convert('RGB')
img = img.resize((224, 224))

x = np.array(img).astype(np.float32)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)[0]

class_idx = np.argmax(preds)
class_name = labels[class_idx]
accuracy = preds[class_idx]

print(f"Prediction: {class_name} ({accuracy*100:.2f}%)")


# =========================
# RESNET INFERENCE (OPTIONAL)
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = models.resnet18(pretrained=False)
resnet.fc = nn.Linear(resnet.fc.in_features, 4)
resnet.load_state_dict(torch.load('resnet_model.pt', map_location=device))
resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class_labels = ['both', 'crackles', 'healthy', 'wheezes']

image = Image.open(output_filename).convert('RGB')
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = resnet(image)
    _, predicted = torch.max(outputs.data, 1)

print("ResNet Prediction:", class_labels[predicted.item()])