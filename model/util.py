import numpy as np
import math
import torch
import os
import sys
import time
from torch import inf
import wandb
from sre_constants import CATEGORY
import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from einops.layers.torch import Rearrange
import math
import json
import torch
import re
import open_clip
from torch.utils.data import DataLoader
from natsort import natsorted
import random
import matplotlib.pyplot as plt

import sys
sys.path.append("/mnt/dataset0/kyw/closed-loop")

from CORnet.cornet import CORnet_S


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # Load model, training preprocessor and feature extractor
# vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
#     model_name = 'ViT-H-14', pretrained = None, precision='fp32', device=device
# )

# # Load the previously downloaded weight file
# model_weights_path = "/mnt/repo0/kyw/open_clip_pytorch_model.bin"
# model_state_dict = torch.load(model_weights_path, map_location=device)
# vlmodel.load_state_dict(model_state_dict)

# # Set model to evaluation mode
# vlmodel.eval()

# EEG signal generation module
# Define your model structure
# def create_model(device):
#     model = models.alexnet(pretrained=False)
#     model.classifier[6] = torch.nn.Linear(4096, 17*250)  # 17 channels x 100 time points = 1700 outputs
#     model = model.to(device)
#     return model

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

def load_model_endocer(model_path, dnn, device):
    model = create_model(device, dnn) #dnn='alexnet' ,'cornet_s'
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['best_model'])
    model.eval()
    return model



# def load_model_endocer(model_path, device):
#     model = create_model(device, 'alexnet')
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint['best_model'])
#     model.eval()
#     return model
# Load model weights
# def load_model_endocer(model_path, device):
#     model = create_model(device)  # First create the model
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint['best_model'])  # Load model parameters
#     model.eval()
#     return model

# Image preprocessing function
def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Model accepts 224x224 image input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    return image_tensor

# EEG signal generation function
def generate_eeg(model, image_tensor, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        # Feed image into model to generate EEG signal
        eeg_output = model(image_tensor).detach().cpu().numpy()
        
        # Assuming model output is a (1, 1700) vector, reshape to (17, 100)
        eeg_output = np.reshape(eeg_output, (17, 250))
    
    return eeg_output

# Save EEG signal function
def save_eeg_signal(eeg_signal, save_dir, idx, category):
    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_name = f"{category}_{idx + 1}.npy"  # Filename includes sortable identifier
    file_path = os.path.join(save_dir, file_name)
        
    # Save EEG signal to .npy file
    np.save(file_path, eeg_signal)
    # print(f"EEG signal saved to: {file_path}")


def plot_similarity_range(similarities_per_iteration, save_folder):
    """
    Plot the similarity range for each iteration.
    :param similarities_per_iteration: List of similarities per iteration, shape [iterations, num_images_per_iteration]
    :param save_folder: Folder to save the plot
    """
    plt.figure(figsize=(10, 6))

    num_iterations = len(similarities_per_iteration)  # Number of iterations
    iterations = range(1, num_iterations + 1)

    # Calculate average, max, and min similarity for each iteration
    avg_similarities = [np.mean(similarities) for similarities in similarities_per_iteration]
    max_similarities = [np.max(similarities) for similarities in similarities_per_iteration]
    min_similarities = [np.min(similarities) for similarities in similarities_per_iteration]

    # Plot error band (range-based error band)
    plt.fill_between(iterations, min_similarities, max_similarities, color='lightblue', alpha=0.5, label='Range (Min-Max)')

    # Plot mean similarity line
    plt.plot(iterations, avg_similarities, color='blue', label='Average Similarity', linewidth=2)

    # Set border line width
    plt.gca().spines['top'].set_linewidth(0.8)  # Top border
    plt.gca().spines['right'].set_linewidth(0.8)  # Right border
    plt.gca().spines['left'].set_linewidth(0.8)  # Left border
    plt.gca().spines['bottom'].set_linewidth(0.8)  # Bottom border

    # Set labels and legend
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Similarity', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)

    # Display and save the figure
    plt.tight_layout()

    save_path = os.path.join(save_folder, "Similarity_Plot.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)  # Save as high-resolution image
    print(f"Visualization saved to {save_path}")

    # Show and close the figure
    plt.show()
    plt.close()


def save_value_function_to_txt(value_function, save_folder,iteration):
    """
    Save value_function to a .txt file with 12 values per line.
    :param value_function: Values to save (PyTorch tensor or NumPy array)
    :param save_folder: Folder path to save the file
    """
    # Convert to NumPy array if value_function is a PyTorch tensor
    if isinstance(value_function, torch.Tensor):
        value_function = value_function.cpu().numpy()

    # Build save path
    save_path = os.path.join(save_folder, f'value_function_scores_{iteration}.txt')

    # Write values to .txt file (12 per line)
    with open(save_path, 'w') as f:
        for i in range(0, len(value_function), 12):
            line_values = value_function[i:i + 12]
            line_str = " ".join(map(str, line_values))  # Space-separated values
            f.write(f"{line_str}\n")  # Write 12 values per line
    
    print(f"value_function saved to {save_path}")


def save_amx_similarities(folder_path, loooop_max_similarities):
    """
    Save images, similarities, and variance for each iteration.
    :param folder_path: Root folder for saving
    :param iteration: Current iteration round
    :param image_paths: List of image paths
    :param similarities: List of similarities
    :param variance: Variance of similarities
    """
    iter_folder = os.path.join(folder_path)
    os.makedirs(iter_folder, exist_ok=True)
    
    # # Save images
    # for i, image_path in enumerate(image_paths):
    #     image = Image.open(image_path)
    #     image.save(os.path.join(iter_folder, f"selected_image_{i}.jpg"))
    
    # Save similarities and variance
    with open(os.path.join(iter_folder, "similarities.txt"), "w") as f:
        f.write(f"MAx similarity: {loooop_max_similarities}\n")
        # # f.write(f"Variance: {variance}\n")
        # f.write(f"Similarities: {similarities}\n")

def save_results(folder_path, iteration, image_paths,  similarities, max_similarity):
    """
    Save images, similarities, and variance for each iteration.
    :param folder_path: Root folder for saving
    :param iteration: Current iteration round
    :param image_paths: List of image paths
    :param similarities: List of similarities
    :param variance: Variance of similarities
    """
    iter_folder = os.path.join(folder_path, f"iteration_{iteration}")
    os.makedirs(iter_folder, exist_ok=True)
    
    # Save images
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        image.save(os.path.join(iter_folder, f"selected_image_{i}.jpg"))
    
    # Save similarities and variance
    with open(os.path.join(iter_folder, "similarities.txt"), "w") as f:
        f.write(f"similarities: {similarities}\n")
        f.write(f"MAx similarity: {max_similarity}\n")
        # f.write(f"Variance: {variance}\n")
        f.write(f"Similarities: {similarities}\n")
        
def get_image_path(category_idx, image_idx, text_list):
    """
    Return image path based on category and image index, checking if folder contains 12 images.
    :param category_idx: Category index
    :param image_idx: Image index
    :param text_list: List of folders
    :return: Path to the image
    """
    # Get category folder path

    category_folder = text_list[category_idx]
    # print(category_idx)
    # print(category_folder)
    folder_path = f"/mnt/dataset0/kyw/closed-loop/image_select/{category_folder}"
    
    image_file = [f for f in sorted(os.listdir(folder_path)) if f.endswith(('.jpg', '.png')) and not f.startswith('._')]
    
    # print(image_file)
        # # Check if the folder contains exactly 12 images
    # images_in_folder = sorted(os.listdir(folder_path))
    
    
    # if len(images_in_folder) != 12:
    #     raise ValueError(f"Error: The folder {folder_path} contains {len(images_in_folder)} images, but 12 are expected.")
    
    # # Get image filenames, assuming images in the folder are sorted in some order
    image_file = image_file[image_idx]
    
    # Return the full image path
    return os.path.join(folder_path, image_file)

def load_thingstestimagedata(img_directory):
    images = []
    category_images = []  # Stores 12 images per category
    all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
    all_folders.sort()  # Ensure consistent folder ordering
    # print(all_folders)
    for folder in all_folders:
        # print(folder)
        folder_path = os.path.join(img_directory, folder)
        folder_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        folder_images.sort()  # Ensure images within each category are ordered
        # print(folder_images)
        
        
        # Keep only the first 12 images
        folder_images = folder_images[:12]
        full_image_paths = [os.path.join(folder_path, img) for img in folder_images]
        category_images.append(full_image_paths)
       
    # print(category_images)
    return category_images  # Return list of image paths per category

def save_eeg(synthetic_eeg, gt_eeg_path, file_name):
# Filename includes sortable identifier
    gt_eeg_path = os.path.join(gt_eeg_path, file_name)
    np.save(gt_eeg_path, synthetic_eeg)
    return gt_eeg_path

def get_gteeg(image_gt_path, encoder_model_path,dnn, device):
    # img_model.eval()

    model = load_model_endocer(encoder_model_path,dnn, device)
    image_tensor = preprocess_image(image_gt_path, device)
    synthetic_eeg = generate_eeg(model, image_tensor, device)
    
    # save_eeg_signal(synthetic_eeg, syn_eeg_path, idx=0, category = category)
    return synthetic_eeg

def plot_similarity_and_mse_with_dual_axis(similarities_per_iteration, save_folder, target_similarity=1.0):
    """
    Plot similarity and MSE curves with error bands using dual y-axes, with adjusted border line width.
    
    Args:
        similarities_per_iteration (list): List of similarities per iteration, shape [iterations, num_images_per_iteration]
        save_folder (str): Folder path to save the plot
        target_similarity (float, optional): Target similarity for MSE computation, default 1.0 (perfect similarity)
    """
    # Data preprocessing
    num_iterations = len(similarities_per_iteration)
    iterations = range(1, num_iterations + 1)
    
    # Compute statistics
    avg_similarities = [np.mean(s) for s in similarities_per_iteration]
    max_similarities = [np.max(s) for s in similarities_per_iteration]
    min_similarities = [np.min(s) for s in similarities_per_iteration]
    
    # Compute MSE (based on max_similarities)
    mse_per_iteration = [(sim - target_similarity)**2 for sim in max_similarities]
    
    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot similarity curve (left axis)
    ax1.fill_between(iterations, min_similarities, max_similarities, 
                    color='lightblue', alpha=0.3, label='Similarity Range')
    ax1.plot(iterations, avg_similarities, color='blue', 
            linewidth=2, label='Avg Similarity')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Similarity', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Create right axis and plot MSE curve
    ax2 = ax1.twinx()
    ax2.plot(iterations, mse_per_iteration, color='red', 
            linewidth=2, linestyle='--', label='MSE')
    ax2.set_ylabel('MSE', fontsize=12)
    
    # Style adjustments
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper center', fontsize=10, ncol=3,
              bbox_to_anchor=(0.5, 1.15))
    
    # Save and display
    plt.tight_layout()
    save_path = os.path.join(save_folder, "Similarity_and_MSE_Dual_Axis.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved to: {save_path}")
    plt.show()
    plt.close()
    
    
def visualize_images(image_paths, save_folder, iteration):
    """
    Display selected images in order using matplotlib.
    :param image_paths: List of image paths
    :param similarities: Similarity value for each image
    """


    # Plot images
    fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 5))
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        axes[i].imshow(image)
        axes[i].axis('off')
                
    plt.show()
    
    os.makedirs(save_folder, exist_ok=True)  # Create folder if it doesn't exist
    save_path = os.path.join(save_folder, f"visualization_iteration_{iteration}.png")
    fig.savefig(save_path, bbox_inches='tight')  # Save image file
    print(f"Visualization saved to {save_path}")

        
def visualize_top_images(image_paths, similarities, save_folder, iteration):
    """
    Display selected images sorted by similarity in descending order using matplotlib.
    :param image_paths: List of image paths
    :param similarities: Similarity value for each image
    """
    # Pair image paths with similarities and sort by similarity in descending order
    image_similarity_pairs = sorted(zip(image_paths, similarities), key=lambda x: x[1], reverse=True)
    
    # Unzip sorted image paths and similarities
    sorted_image_paths, sorted_similarities = zip(*image_similarity_pairs)

    # Plot images
    fig, axes = plt.subplots(1, len(sorted_image_paths), figsize=(15, 5))
    for i, image_path in enumerate(sorted_image_paths):
        image = Image.open(image_path)
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(f'Similarity: {sorted_similarities[i]:.4f}', fontsize=8)  # Display similarity
    plt.show()
    
    os.makedirs(save_folder, exist_ok=True)  # Create folder if it doesn't exist
    save_path = os.path.join(save_folder, f"visualization_iteration_{iteration}.png")
    fig.savefig(save_path, bbox_inches='tight')  # Save image file
    print(f"Visualization saved to {save_path}")

def extract_number(filename):
    """
    Extract numbers from a filename. Returns 0 if no numbers are found.
    
    """
    
    numbers = re.findall(r'(\d+)', filename)
    if numbers:
        return tuple(map(int, numbers))  # Return tuple of multiple numbers
    return (float('inf'),)  # If no numbers found, return a tuple with a very large value

    # match = re.search(r'(\d+)', filename)
    # return int(match.group(1)) if match else 0

# Iterate through image folder and process each image
def generate_and_save_eeg_for_all_images(model_path, test_image_list, save_dir, device, category_list):
    model = load_model_endocer(model_path, device)
    for idx, image_path in enumerate(test_image_list):
        image_tensor = preprocess_image(image_path, device)
        synthetic_eeg = generate_eeg(model, image_tensor, device)
        category = category_list[idx]
        save_eeg_signal(synthetic_eeg, save_dir, idx, category)

    # # Load the previously saved model
    # # model_path = '/mnt/repo0/kyw/close-loop/sub_model/sub-08/generation/encoding-end_to_end/dnn-alexnet/modeled_time_points-all/pretrained-False/model_state_dict_250hz.pt'
    # model = load_model_endocer(model_path, device)
    #  # EEG signal save path

    # # Iterate through each image in the image folder
    # # image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))])
    # # image_files = natsorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png')) and not f.startswith('._')])
    # # image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png')) and not f.startswith('._')]
    # # image_files.sort(key=extract_number)
    # image_files = [img for img in test_image_list]
    # image_files.sort(key=lambda x: extract_number(x))
    # # image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png')) and not f.startswith('._')])


    # for idx, image_file in enumerate(image_files):
    #     image_path = os.path.join(image_folder, image_file)
    #     # print(f"Processing image: {image_path}")

    #     # Preprocess image
    #     image_tensor = preprocess_image(image_path, device)

    #     # Generate EEG signal
    #     synthetic_eeg = generate_eeg(model, image_tensor, device)

    #     # Save EEG signal
    #     save_eeg_signal(synthetic_eeg, save_dir, idx, image_gen)
        

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute one extra element to accommodate odd dimensions
        div_term = torch.exp(torch.arange(0, d_model + 1, 2).float() * (-math.log(10000.0) / d_model))
        # Use slicing to prevent overflow
        pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2 + 1])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1)
        x = x + pe
        return x

class EEGAttention(nn.Module):
    def __init__(self, channel, d_model, nhead):
        super(EEGAttention, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.channel = channel
        self.d_model = d_model

    def forward(self, src):
        src = src.permute(2, 0, 1)  # Change shape to [time_length, batch_size, channel]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output.permute(1, 2, 0)  # Change shape back to [batch_size, channel, time_length]
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 5), (1, 1)),
            nn.AvgPool2d((1, 17), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (17, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.tsconv(x)
        x = self.projection(x)
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x

class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1840, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class ATM_S_reconstruction_scale_0_1000(nn.Module):    
    def __init__(self, num_channels=17, sequence_length=250, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
        super(ATM_S_reconstruction_scale_0_1000, self).__init__()
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)   
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  
         
    def forward(self, x):
        x = self.attention_model(x)
        # print(f'After attention shape: {x.shape}')
         
        x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        eeg_embedding = self.enc_eeg(x)
        # print(f'After enc_eeg shape: {eeg_embedding.shape}')
        out = self.proj_eeg(eeg_embedding)
        return out  
    
def load_model_decoder(model_path, device):
    """
    Load pretrained EEG and image models along with optimizer state.
    """
    checkpoint = torch.load(model_path, map_location=device)
    eeg_model = ATM_S_reconstruction_scale_0_1000(17, 250)  # e.g. ATM_S_reconstruction_scale_0_1000 as the EEG model
    img_model = Proj_img()  # Assuming Proj_img is the image model

    eeg_model.load_state_dict(checkpoint['eeg_model_state_dict'])
    img_model.load_state_dict(checkpoint['img_model_state_dict'])
    optimizer_state = checkpoint['optimizer_state_dict']
    
    return eeg_model.to(device), img_model.to(device), optimizer_state

def ImageEncoder(images, img_model, preprocess_train, device):
    batch_size = 20  # Set to an appropriate value
    image_features_list = []
      
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        image_inputs = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images]).to(device)

        with torch.no_grad():
            batch_image_features = vlmodel.encode_image(image_inputs)
            # batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)

        image_features_list.append(batch_image_features)

    image_features = torch.cat(image_features_list, dim=0)
        
    return image_features

def load_all_eeg_signals(eeg_folder):
    """
    Load EEG signals from a folder.
    """
    eeg_paths = []
    for filename in os.listdir(eeg_folder):
        if filename.endswith('.npy'):
            eeg_paths.append(os.path.join(eeg_folder, filename))
    eeg_signals = []

    for path in eeg_paths:
        # Check if file exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

        # Load a single EEG signal
        eeg_tensor = load_eeg_signals(path)  # Shape: [17, 100]

        eeg_signals.append(eeg_tensor)

    # Stack all EEG signals into a tensor of shape [N, 17, 100]
    combined_eeg = torch.stack(eeg_signals, dim=0)  # Shape: [81, 17, 100]
    
    return  combined_eeg

def load_eeg_signals(eeg_path):

    eeg_data = np.load(eeg_path)
    eeg_signals = torch.tensor(eeg_data)
    return eeg_signals

def get_eeg_features(eeg_model, eeg_signal, device):
    eeg_model.eval()
    eeg_model.to(device)
    eeg_signal = eeg_signal.to(device)
    subject_ids = torch.full((1,), int('08'), dtype=torch.long).to(device)  
    eeg_embeds = eeg_model(eeg_signal.unsqueeze(0), subject_ids).float()  # Pass EEG signal through the model

    return eeg_embeds

def get_img_features(img_model, preprocess_train,vlmodel,device,img_path):
    img_model.to(device)
    img_model.eval()
    image_input = torch.stack([preprocess_train(Image.open(img_path).convert("RGB"))]).to(device)
    img_embeds = vlmodel.encode_image(image_input)
    return img_embeds
    
def evaluate_eeg_signals(eeg_model, img_model, eeg_signals_truth, device, truth_folder, false_folder, truth, false):
    """
    Classify given EEG signals and compute accuracy.
    """
    correct = 0
    total = 0
    correct_samples = []

    # Load images from truth and false folders and generate features
    img_truth_paths = [os.path.join(truth_folder, f"{truth}_{i+1}.jpg") for i in range(len(eeg_signals_truth))]
    img_truth = ImageEncoder(img_truth_paths, img_model, preprocess_train, device)

    img_false_paths = [os.path.join(false_folder, f"{false}_{i+1}.jpg") for i in range(1)]
    img_false = ImageEncoder(img_false_paths, img_model, preprocess_train, device)

    with torch.no_grad():
        # Iterate through EEG signals in truth and false folders
        for idx, eeg_data in enumerate(eeg_signals_truth):
            eeg_data = eeg_data.to(device)
            eeg_features = eeg_model(eeg_data.unsqueeze(0)).float()  # Pass EEG signal through the model
            logit_scale = eeg_model.logit_scale

            # Extract truth image features
            img_features_truth = img_truth[idx].unsqueeze(0).float()
            img_features_false = img_false[idx % len(img_false)].unsqueeze(0).float()  # Cycle through false features

            # Compute logits against truth and false images
            logits_truth = logit_scale * (eeg_features @ img_features_truth.T)
            logits_false = logit_scale * (eeg_features @ img_features_false.T)

            # Concatenate truth and false logits to determine classification
            logits = torch.cat([logits_truth, logits_false], dim=1)
            predicted_label = torch.argmax(logits)

            true_label = 0  # Label for truth is 0

            if predicted_label == true_label:
                correct += 1
                correct_samples.append(idx)

            total += 1

        # Iterate through EEG signals in false folder
        # for idx, eeg_data in enumerate(eeg_signals_false):
        #     eeg_data = eeg_data.to(device)
        #     eeg_features = eeg_model(eeg_data.unsqueeze(0)).float()  # Pass EEG signal through the model
        #     logit_scale = eeg_model.logit_scale

        #     # Extract truth and false image features
        #     img_features_truth = img_truth[idx % len(img_truth)].unsqueeze(0).float()  # Cycle through truth features
        #     img_features_false = img_false[idx].unsqueeze(0).float()

        #     # Compute logits against truth and false images
        #     logits_truth = logit_scale * (eeg_features @ img_features_truth.T)
        #     logits_false = logit_scale * (eeg_features @ img_features_false.T)

        #     # Concatenate truth and false logits to determine classification
        #     logits = torch.cat([logits_truth, logits_false], dim=1)
        #     predicted_label = torch.argmax(logits)

        #     true_label = 1  # Label for false is 1

        #     if predicted_label == true_label:
        #         correct += 1
        #         correct_samples.append(idx + len(eeg_signals_truth))  # Offset index by the number of truth samples

        #     total += 1

    accuracy = correct / total
    return accuracy, correct_samples




def classification(gene_image_embed, gene_eeg_embed, num_class, test_image_embeds, idx):
    num_test_samples = test_image_embeds.shape[0]
    all_idxes = list(range(num_test_samples))
    rest_idxes = [i for i in all_idxes if i != idx]

    select_idxes = random.sample(rest_idxes, num_class - 1)

    similarities = {}
    for select_idx in select_idxes:
        similarity = F.cosine_similarity(gene_eeg_embed, test_image_embeds[select_idx])
        similarities[select_idx] = similarity

    gene_sim = F.cosine_similarity(gene_image_embed, gene_eeg_embed)
    similarities[idx] = gene_sim

    max_idx = max(similarities, key=similarities.get)

    if max_idx == idx:
        return 1
    else:
        return 0

def get_image_pool(image_set_path):
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
    return test_images_path, labels 

def sample_from_image_pool(image_paths, labels, k):
    idxes = random.sample(range(len(image_paths)), k)
    return [image_paths[idx] for idx in idxes], [labels[idx] for idx in idxes]





class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
        


def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm
        
def train_one_epoch(model, data_loader, optimizer, device, epoch, 
                        loss_scaler, log_writer=None, config=None, start_time=None, model_without_ddp=None, 
                        img_feature_extractor=None, preprocess=None):
    model.train(True)
    optimizer.zero_grad()
    total_loss = []
    total_cor = []
    accum_iter = config.accum_iter
    for data_iter_step, (data_dcit) in enumerate(data_loader):
        
        # we use a per iteration (instead of per epoch) lr scheduler
        # print(data_iter_step)
        # print(len(data_loader))
        
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)
        samples = data_dcit['eeg']
        
        img_features = None
        valid_idx = None
        if img_feature_extractor is not None:
            images = data_dcit['image']
            valid_idx = torch.nonzero(images.sum(dim=(1,2,3)) != 0).squeeze(1)
            img_feature_extractor.eval()
            with torch.no_grad():
                img_features = img_feature_extractor(preprocess(images[valid_idx]).to(device))['layer2']
        samples = samples.to(device)
        # img_features = img_features.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            loss, pred, _ = model(samples, img_features, valid_idx=valid_idx, mask_ratio=config.mask_ratio)
        # loss.backward()
        # norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        # optimizer.step()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}")
            sys.exit(1)

        # loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad)

        # if (data_iter_step + 1) % accum_iter == 0:
        # cal the cor
        pred = pred.to('cpu').detach()
        samples = samples.to('cpu').detach()
        # pred = pred.transpose(1,2) #model_without_ddp.unpatchify(pred)
        pred = model_without_ddp.unpatchify(pred)
            
        cor = torch.mean(torch.tensor([torch.corrcoef(torch.cat([p[0].unsqueeze(0), s[0].unsqueeze(0)],axis=0))[0,1] for p, s in zip(pred, samples)])).item()
        optimizer.zero_grad()

        total_loss.append(loss_value)
        total_cor.append(cor)
        if device == torch.device('cuda:0'):
            lr = optimizer.param_groups[0]["lr"]
            print('train_loss_step:', np.mean(total_loss), 'lr:', lr, 'cor', np.mean(total_cor))

    if log_writer is not None:
        lr = optimizer.param_groups[0]["lr"]
        log_writer.log('train_loss_step', np.mean(total_loss), step=epoch)
        log_writer.log('lr', lr, step=epoch)
        log_writer.log('cor', np.mean(total_cor), step=epoch)
        if start_time is not None:
            log_writer.log('time (min)', (time.time() - start_time)/60.0, step=epoch)
    if config.local_rank == 0:        
        print(f'[Epoch {epoch}] loss: {np.mean(total_loss)}')

    return np.mean(total_cor)

def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_l = np.arange(length, dtype=float)

    grid_l = grid_l.reshape([1, length])
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_l)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches # cls token
        # height (== width) for the checkpoint position embedding
        orig_size = int(pos_embed_checkpoint.shape[-2] - num_extra_tokens)
        # height (== width) for the new position embedding
        new_size = int(num_patches)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %d to %d" % (orig_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, embedding_size).permute(0, 2, 1)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size))
            pos_tokens = pos_tokens.permute(0, 2, 1)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < config.warmup_epochs:
        lr = config.lr * epoch / config.warmup_epochs 
    else:
        lr = config.min_lr + (config.lr - config.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - config.warmup_epochs) / (config.num_epoch - config.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


    

def load_model(config, model, checkpoint_path ):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print(f'Model loaded with {checkpoint_path}')
    

def patchify(imgs, patch_size):
    """
    imgs: (N, 1, num_voxels)
    x: (N, L, patch_size)
    """
    p = patch_size
    assert imgs.ndim == 3 and imgs.shape[2] % p == 0

    h = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], h, p))
    return x

def unpatchify(x, patch_size):
    """
    x: (N, L, patch_size)
    imgs: (N, 1, num_voxels)
    """
    p = patch_size
    h = x.shape[1]
    
    imgs = x.reshape(shape=(x.shape[0], 1, h * p))
    return imgs

class wandb_logger:
    def __init__(self, config):
        try:
            wandb.init(
                # Set the project where this run will be logged
                project=config['project'],
                name=config['name'],
                config=config,
                entity=config['entity'],            
                )
        except:
                wandb.init(
                # Set the project where this run will be logged
                project=config.project,
                name=config.name,
                config=config,
                entity=config.entity,            
                )

        self.config = config
        self.step = None
    
    def log(self, data, step=None):
        if step is None:
            wandb.log(data)
        else:
            wandb.log(data, step=step)
            self.step = step
    
    def watch_model(self, *args, **kwargs):
        wandb.watch(*args, **kwargs)

    def log_image(self, figs):
        if self.step is None:
            wandb.log(figs)
        else:
            wandb.log(figs, step=self.step)

    def finish(self):
        wandb.finish(quiet=True)

    def load(self, net):
        path = os.path.join(self.config['path_data'], self.config['path_ckpt'], self.config['file_ckpt'])
        net.load_state_dict(torch.load(path))
        print(f'load {path}')

    def save(self, net, file_name=None):
        path_ckpt = os.path.join(self.config['path_data'], self.config['path_ckpt'])
        if not os.path.exists(path_ckpt):
            os.makedirs(path_ckpt)
            print(f'{path_ckpt} created!')

        path = os.path.join(path_ckpt, file_name)
        torch.save(net.state_dict(), path)

    def watch(self, model, log):
        wandb.watch(model, log)