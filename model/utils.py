import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
# from cornet.cornet_s import CORnet_S
from torchvision import models

def create_model(device, dnn):
    if dnn == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = torch.nn.Linear(4096, 4250) 
    if dnn == 'cornet_s':
        model = CORnet_S()
        model.decoder = nn.Sequential(
            model.decoder.avgpool,
            model.decoder.flatten,
            model.decoder.linear,
            nn.Linear(in_features=1000, out_features=4250), 
            model.decoder.output 
        )
    model = model.to(device)
    return model

def load_model_encoder(model_path, device):
    model = create_model(device, 'alexnet')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['best_model'])
    model.eval()
    return model

def preprocess_image(path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    
    image = Image.open(path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def generate_eeg(model, image_tensor, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        eeg_output = model(image_tensor).detach().cpu().numpy()
        eeg_output = np.reshape(eeg_output, (17, 250))
    return eeg_output

def save_eeg_signal(eeg_signal, save_dir, idx, category):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = f"{category}_{idx + 1}.npy" 
    file_path = os.path.join(save_dir, file_name)
    np.save(file_path, eeg_signal)

