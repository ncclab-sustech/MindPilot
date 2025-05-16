import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import random
from scipy.special import softmax
from mne.time_frequency import psd_array_multitaper
from model.utils import preprocess_image, generate_eeg
import pickle
def load_vlmodel(model_name='ViT-H-14', model_weights_path=None, precision='fp32', device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
        model_name=model_name, pretrained=None, precision=precision, device=device
    )
    if model_weights_path:
        model_state_dict = torch.load(model_weights_path, map_location=device)
        vlmodel.load_state_dict(model_state_dict)
    vlmodel.eval()
    return vlmodel, preprocess_train, feature_extractor

# def get_image_pool(image_set_path):
#     test_images_path = []
#     labels = []
#     for sub_test_image in sorted(os.listdir(image_set_path)):
#         if sub_test_image.startswith('.'):
#             continue
#         sub_image_path = os.path.join(image_set_path, sub_test_image)
#         for image in sorted(os.listdir(sub_image_path)):
#             if image.startswith('.'):
#                 continue
#             image_label = os.path.splitext(image)[0]
#             labels.append(image_label)
#             image_path = os.path.join(sub_image_path, image)
#             test_images_path.append(image_path)
#     return test_images_path, labels

def get_image_pool(image_set_path, cache_file='image_paths_cache.pkl'):
    # 检查是否有缓存文件
    if os.path.exists(cache_file):
        print(f"Loading cached image paths from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # 如果没有缓存，则扫描目录
    test_images_path = []
    labels = []
    for sub_test_image in sorted(os.listdir(image_set_path)):
        if sub_test_image.startswith('.'):
            continue
        sub_image_path = os.path.join(image_set_path, sub_test_image)
        for image in sorted(os.listdir(sub_image_path)):
            if image.startswith('.'):
                continue
            image_label = os.path.splitext(image)[0]
            labels.append(image_label)
            image_path = os.path.join(sub_image_path, image)
            test_images_path.append(image_path)
    
    # 保存结果到缓存文件
    with open(cache_file, 'wb') as f:
        pickle.dump((test_images_path, labels), f)
    
    return test_images_path, labels



# def calculate_loss(eeg_path, target_psd, fs, selected_channel_idxes):
#     eeg = np.load(eeg_path, allow_pickle=True)
#     selected_eeg = eeg[selected_channel_idxes, :]
#     psd, _ = psd_array_multitaper(selected_eeg, fs, adaptive=True, normalization='full', verbose=0)
#     psd = torch.from_numpy(psd.flatten()).unsqueeze(0)
#     target_psd = torch.tensor(target_psd).view(1, 378)
#     loss_fn = nn.MSELoss()
#     loss = loss_fn(psd, target_psd)
#     return loss



def load_target_psd(target_path, fs, selected_channel_idxes):
    target_signal = np.load(target_path, allow_pickle=True)
    selected_target_signal = target_signal[selected_channel_idxes, :]
    target_psd, _ = psd_array_multitaper(selected_target_signal, fs, adaptive=True, normalization='full', verbose=0)
    return torch.from_numpy(target_psd.flatten()).unsqueeze(0)





def get_target_eeg(model_path, target_image_path, save_dir, device):
    model = load_model_endocer(model_path, device)
    image_tensor = preprocess_image(target_image_path, device)
    synthetic_eeg = generate_eeg(model, image_tensor, device)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.splitext(os.path.basename(target_image_path))[0]
    target_eeg_path = os.path.join(save_dir, f"{filename}.npy")
    np.save(target_eeg_path, synthetic_eeg)
    return target_eeg_path