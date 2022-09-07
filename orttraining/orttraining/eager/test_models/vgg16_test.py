# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# pylint: disable=missing-docstring
# pylint: disable=C0103
# pylint: disable=R0903

# The following is a simple neural network trained and tested using FashinMINST data.
# It is using eager mode targeting the ort device. After building eager mode run
# PYTHONPATH=~/{repo root}/build/Linux/Debug python
# ~/{repo root}/orttraining/orttraining/eager/test_models/vgg16_test.py

import os
from statistics import mode

import onnxruntime_pybind11_state as torch_ort
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.io import read_image
from torchvision.models import VGG16_Weights, vgg16
from torchvision.transforms import ToTensor

device = torch_ort.device()
# torch_ort.set_default_logger_severity(0)
torch_ort.set_default_logger_verbosity(4)


img = read_image("/home/wil/torchPlaying/dog1.jpg")  # "test/assets/encode_jpeg/grace_hopper_517x606.jpg")
# img.to(device)
# Step 1: Initialize model with the best available weights
weights = VGG16_Weights.DEFAULT
model = vgg16(weights=weights).to(device)
model.eval()
# model.to(device)

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

batch.to(device)
print(f"vgg batch device: {batch.device}")

model.to(device)
# Step 4: Use the model and print the predicted category
prediction = model(batch)
prediction = prediction.squeeze(0).softmax(0)
print(f"vgg prediction device: {prediction.device}")
class_id = prediction.argmax().item()
print(f"class id: {class_id}")
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")
