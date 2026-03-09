"""
Benchmark for Heuristic Generation with Three Different Methods:
1. EEG Feature Guidance - Uses target EEG features as the optimization objective
2. Target Image CLIP Guidance - Uses the target image's CLIP embedding as the optimization objective
3. Random Generation - Fully random generation without any optimization (baseline)

Evaluation criteria:
All methods are evaluated using the same metrics:
- EEG Score: EEG feature similarity between generated images and target images
- CLIP Score: CLIP embedding similarity between generated images and target images
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

# proxy = "10.16.11.87:7890"
# os.environ['http_proxy'] = proxy
# os.environ['https_proxy'] = proxy

import numpy as np
import torch
import random
from PIL import Image
from scipy.special import softmax
import open_clip
from mne.time_frequency import psd_array_multitaper
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/home/ldy/Workspace/Closed_loop_optimizing')
sys.path.append('/home/ldy/Workspace/Closed_loop_optimizing/model')

from model.utils import load_model_encoder, generate_eeg, save_eeg_signal
from model.custom_pipeline_low_level import Generator4Embeds
from model.ATMS_retrieval import ATMS, get_eeg_features
from model.pseudo_target_model import PseudoTargetModel
import einops
import logging

logging.getLogger("diffusers").setLevel(logging.WARNING)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Configure image and embedding directories
image_dir = '/home/ldy/Workspace/Closed_loop_optimizing/test_images'
embed_dir = '/home/ldy/Workspace/Closed_loop_optimizing/data/clip_embed/2025-09-21'

# Get all image and embedding paths, sorted to ensure one-to-one correspondence
image_list = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
embed_list = sorted([f for f in os.listdir(embed_dir) if f.endswith('_embed.pt')])


class HeuristicGenerator:
    def __init__(self, pipe, vlmodel, preprocess_train, device="cuda", seed=42, load_ip_adapter=True, min_data_threshold=10):
        self.pipe = pipe
        self.vlmodel = vlmodel
        self.preprocess_train = preprocess_train
        self.device = device

        # Hyperparameters
        self.batch_size = 32
        self.alpha = 80
        self.total_steps = 15
        self.max_inner_steps = 10
        self.num_inference_steps = 8
        self.guidance_scale = 0.0
        self.dimension = 1024
        self.self_improvement_ratio = 0.5
        self.reward_scaling_factor = 100
        self.initial_step_size = 30
        self.decay_rate = 0.1
        self.generate_batch_size = 1
        self.save_per = 5
        self.min_data_threshold = min_data_threshold  # minimum data count threshold

        # Initialize components
        self.pseudo_target_model = PseudoTargetModel(dimension=self.dimension, noise_level=1e-4).to(self.device)
        self.generator = torch.Generator(device=device).manual_seed(seed)

        # Load IP adapter only if requested (to avoid repeated loading)
        if load_ip_adapter:
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter", subfolder="sdxl_models", 
                weight_name="ip-adapter_sdxl_vit-h.bin", 
                torch_dtype=torch.bfloat16)
            self.pipe.set_ip_adapter_scale(0.5)

    def reward_function_embed(self, embed1, embed2):
        """Compute reward based on cosine similarity between CLIP embeddings"""
        cosine_sim = F.cosine_similarity(embed1, embed2, dim=1)
        normalized_sim = (cosine_sim + 1) / 2
        return normalized_sim

    def latents_to_images(self, latents):
        shift_factor = self.pipe.vae.config.shift_factor if self.pipe.vae.config.shift_factor else 0.0
        latents = (latents / self.pipe.vae.config.scaling_factor) + shift_factor
        images = self.pipe.vae.decode(latents, return_dict=False)[0]
        images = self.pipe.image_processor.postprocess(images.detach())
        return images

    def x_flatten(self, x):
        return einops.rearrange(x, '... C W H -> ... (C W H)', 
                            C=self.pipe.unet.config.in_channels, 
                            W=self.pipe.unet.config.sample_size, 
                            H=self.pipe.unet.config.sample_size)

    def x_unflatten(self, x):
        return einops.rearrange(x, '... (C W H) -> ... C W H', 
                            C=self.pipe.unet.config.in_channels, 
                            W=self.pipe.unet.config.sample_size, 
                            H=self.pipe.unet.config.sample_size)

    def get_norm(self, epsilon):
        return self.x_flatten(epsilon).norm(dim=-1)[:,:,None,None,None]

    def merge_images_grid(self, image_grid):
        rows = len(image_grid)
        cols = len(image_grid[0])
        img_width, img_height = image_grid[0][0].size
        merged_image = Image.new('RGB', (cols * img_width, rows * img_height))

        for row_idx, row in enumerate(image_grid):
            for col_idx, img in enumerate(row):
                merged_image.paste(img, (col_idx * img_width, row_idx * img_height))

        return merged_image

    def generate(self, data_x, data_y, tar_image_embed, prompt='', save_path=None, start_embedding=None):
        # Initialize noise
        epsilon = torch.randn(self.num_inference_steps+1, self.generate_batch_size, 
                            self.pipe.unet.config.in_channels, 
                            self.pipe.unet.config.sample_size, 
                            self.pipe.unet.config.sample_size, 
                            device=self.device, generator=self.generator)

        epsilon_init = epsilon.clone()
        epsilon_init_norm = self.get_norm(epsilon_init)
        all_images = []

        img_save_dir = None
        if save_path is not None:
            img_save_dir = os.path.join(save_path, "generated_imgs")
            os.makedirs(img_save_dir, exist_ok=True)
        img_counter = 0

        # Initialize pseudo target
        if start_embedding is not None:
            pseudo_target = start_embedding.expand(self.generate_batch_size, self.dimension).to(self.device)
        else:
            pseudo_target = torch.randn(self.generate_batch_size, self.dimension, device=self.device, generator=self.generator)

        # Optimization loop - only optimize pseudo_target, no image generation
        for step in range(self.total_steps):
            data_x, data_y = self.pseudo_target_model.get_model_data()   
            if data_y.size(0) < self.min_data_threshold:  # check if data count is sufficient
                print(f"[WARNING] Insufficient data ({data_y.size(0)} < {self.min_data_threshold}), returning random generation")
                # Insufficient data, generate random images and return
                latents = self.pipe(
                    [prompt]*self.generate_batch_size,
                    ip_adapter_image_embeds=[pseudo_target.unsqueeze(0).type(torch.bfloat16).to(self.device)],
                    latents=epsilon[0].type(torch.bfloat16),
                    given_noise=epsilon[1:].type(torch.bfloat16),
                    output_type="latent",
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    eta=1.0,
                ).images
                return self.latents_to_images(latents)
            
            # Data sufficient, only perform pseudo_target optimization (no image generation)
            step_size = self.initial_step_size / (1 + self.decay_rate * step)
            pseudo_target, _ = self.pseudo_target_model.estimate_pseudo_target(pseudo_target, step_size=step_size)

        # After optimization loop, generate final images with the optimized pseudo_target
        final_latents = self.pipe(
                [prompt]*self.generate_batch_size,
                ip_adapter_image_embeds=[pseudo_target.unsqueeze(0).type(torch.bfloat16).to(self.device)],
                latents=epsilon[0].type(torch.bfloat16),
                given_noise=epsilon[1:].type(torch.bfloat16),
                output_type="latent",
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                eta=1.0,
            ).images
        
        final_images = self.latents_to_images(final_latents)
        
        # Clean up all intermediate variables from the generate function
        del epsilon, epsilon_init, epsilon_init_norm, pseudo_target, final_latents
        torch.cuda.empty_cache()
        
        return final_images


def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor


def generate_eeg_from_image_paths(model_path, test_image_list, device):
    synthetic_eegs = []
    model = load_model_encoder(model_path, device)
    for idx, image_path in enumerate(test_image_list):
        image_tensor = preprocess_image(image_path, device)
        synthetic_eeg = generate_eeg(model, image_tensor, device)
        synthetic_eegs.append(synthetic_eeg)
    synthetic_eegs = np.asarray(synthetic_eegs)
    return synthetic_eegs


def preprocess_generated_image(image, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor


def generate_eeg_from_image(model_path, images, device):
    synthetic_eegs = []
    model = load_model_encoder(model_path, device)
    for idx, image in enumerate(images):
        image_tensor = preprocess_generated_image(image, device)
        synthetic_eeg = generate_eeg(model, image_tensor, device)
        synthetic_eegs.append(synthetic_eeg)
    return synthetic_eegs


def reward_function_clip_embed_image(pil_image, target_feature, vlmodel, preprocess_train, device):
    """Compute similarity between image CLIP embedding and target feature"""
    tensor_images = [preprocess_train(pil_image)]    
    with torch.no_grad():
        img_embeds = vlmodel.encode_image(torch.stack(tensor_images).to(device))      
    similarity = torch.nn.functional.cosine_similarity(img_embeds.to(device), target_feature.to(device))
    similarity = (similarity + 1) / 2    
    return similarity.item()


def reward_function_clip_embed(eeg, eeg_model, target_feature, sub, device):
    """Compute similarity between EEG feature and target feature"""
    eeg_feature = get_eeg_features(eeg_model, torch.tensor(eeg).unsqueeze(0), device, sub)    
    similarity = torch.nn.functional.cosine_similarity(eeg_feature.to(device), target_feature.to(device))
    similarity = (similarity + 1) / 2
    return similarity.item(), eeg_feature


def fusion_image_to_images(Generator, img_embeds, rewards, device, scale, save_path=None):
    """Image fusion generation"""
    idx1, idx2 = random.sample(range(len(img_embeds)), 2)
    embed1, embed2 = img_embeds[idx1].unsqueeze(0), img_embeds[idx2].unsqueeze(0)
    embed_len = embed1.size(1)
    start_idx = random.randint(0, embed_len - scale - 1)
    end_idx = start_idx + scale
    temp = embed1[:, start_idx:end_idx].clone()
    embed1[:, start_idx:end_idx] = embed2[:, start_idx:end_idx]
    embed2[:, start_idx:end_idx] = temp
    
    generated_images = []        
    with torch.no_grad():         
        images = Generator.generate(img_embeds.to(device), torch.tensor(rewards).to(device), None, prompt='', save_path=save_path, start_embedding=embed1)
        generated_images.extend(images)
        images = Generator.generate(img_embeds.to(device), torch.tensor(rewards).to(device), None, prompt='', save_path=save_path, start_embedding=embed2)
        generated_images.extend(images)
    
    # Return generated images and the indices of the original images used for fusion
    return generated_images, (idx1, idx2)


def select_from_image_paths(probabilities, similarities, losses, sample_image_paths, synthetic_eegs, device, size):
    # Use probability-based sampling during optimization to maintain exploration and diversity
    chosen_indices = np.random.choice(len(probabilities), size=size, replace=False, p=probabilities)
    chosen_similarities = [similarities[idx] for idx in chosen_indices.tolist()] 
    chosen_losses = [losses[idx] for idx in chosen_indices.tolist()]    
    chosen_images = [Image.open(sample_image_paths[i]).convert("RGB") for i in chosen_indices.tolist()]        
    chosen_eegs = [synthetic_eegs[idx] for idx in chosen_indices.tolist()]
    return chosen_similarities, chosen_losses, chosen_images, chosen_eegs


def select_from_images(probabilities, similarities, losses, images_list, eeg_list, size):
    # Use probability-based sampling during optimization to maintain exploration and diversity
    chosen_indices = np.random.choice(len(similarities), size=size, replace=False, p=probabilities)
    chosen_similarities = [similarities[idx] for idx in chosen_indices.tolist()] 
    chosen_losses = [losses[idx] for idx in chosen_indices.tolist()]
    chosen_images = [images_list[idx] for idx in chosen_indices.tolist()]
    chosen_eegs = [eeg_list[idx] for idx in chosen_indices.tolist()]
    return chosen_similarities, chosen_losses, chosen_images, chosen_eegs


def select_top_k_images(similarities, losses, images_list, eeg_list, size):
    """
    Select Top-k images (the k images with the highest reward).
    Used for final evaluation to ensure the best images are selected.
    """
    sorted_indices = np.argsort(similarities)[::-1]  # sort by reward in descending order
    chosen_indices = sorted_indices[:size]  # select the top k
    
    chosen_similarities = [similarities[idx] for idx in chosen_indices.tolist()] 
    chosen_losses = [losses[idx] for idx in chosen_indices.tolist()]
    chosen_images = [images_list[idx] for idx in chosen_indices.tolist()]
    chosen_eegs = [eeg_list[idx] for idx in chosen_indices.tolist()]
    return chosen_similarities, chosen_losses, chosen_images, chosen_eegs


def get_prob_random_sample(test_images_path, model_path, fs, device, selected_channel_idxes, 
                           processed_paths, target_feature, size, method, eeg_model, sub, dnn,
                           vlmodel, preprocess_train):
    """Sample images based on the method and compute rewards"""
    available_paths = [path for path in test_images_path if path not in processed_paths]    
    sample_image_paths = sorted(random.sample(available_paths, min(10, len(available_paths))))
    
    sample_image_name = []
    pil_images = []
    for sample_image_path in sample_image_paths:
        filename = os.path.basename(sample_image_path).split('.')[0]
        sample_image_name.append(filename)    
        pil_images.append(Image.open(sample_image_path).convert("RGB"))
    
    synthetic_eegs = generate_eeg_from_image_paths(model_path, sample_image_paths, device)
    similarities = []
    losses = []
    
    for idx, eeg in enumerate(synthetic_eegs):  
        if method == "eeg_guidance":
            # Method 1: Use EEG feature similarity
            cs, eeg_feature = reward_function_clip_embed(eeg, eeg_model, target_feature, sub, device)
        elif method == "target_image_guidance":
            # Method 2: Use CLIP embedding similarity
            cs = reward_function_clip_embed_image(pil_images[idx], target_feature, vlmodel, preprocess_train, device)
        elif method == "random_generation":
            # Method 3: Random reward (does not affect selection)
            cs = random.random()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        loss = 0  # placeholder
        similarities.append(cs)
        losses.append(loss)

    probabilities = softmax(similarities)
    chosen_similarities, chosen_losses, chosen_images, chosen_eegs = select_from_image_paths(
        probabilities, similarities, losses, sample_image_paths, synthetic_eegs, device, size)
    
    return chosen_similarities, chosen_losses, chosen_images, chosen_eegs


def compute_embed_similarity(img_feature, all_features):
    """Compute cosine similarity between an image and all other images"""
    img_feature = img_feature.float()
    all_features = all_features.float()

    if img_feature.dim() == 1:
        img_feature = img_feature.unsqueeze(0)

    assert torch.isfinite(img_feature).all(), "img_feature contains NaN/Inf values"
    assert torch.isfinite(all_features).all(), "all_features contains NaN/Inf values"    

    img_feature = F.normalize(img_feature, p=2, dim=1)
    all_features = F.normalize(all_features, p=2, dim=1)

    cosine_sim = torch.mm(all_features, img_feature.t()).squeeze(1)
    cosine_sim = (cosine_sim + 1) / 2
    cosine_sim = torch.clamp(cosine_sim, 0.0, 1.0)

    return cosine_sim


def visualize_top_images(images, similarities, save_folder, iteration):
    """Visualize selected images"""
    image_similarity_pairs = sorted(zip(images, similarities), key=lambda x: x[1], reverse=True)
    sorted_images, sorted_similarities = zip(*image_similarity_pairs)

    fig, axes = plt.subplots(1, len(sorted_images), figsize=(15, 5))
    if len(sorted_images) == 1:
        axes = [axes]
    for i, image in enumerate(sorted_images):
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(f'Similarity: {sorted_similarities[i]:.4f}', fontsize=8)
    plt.show()

    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"visualization_iteration_{iteration}.png")
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Visualization saved to {save_path}")
    plt.close()


def get_image_pool(image_set_path):
    test_images_path = []
    for root, dirs, files in os.walk(image_set_path):
        for file in sorted(files):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                test_images_path.append(os.path.join(root, file))
    return test_images_path


def run_single_experiment(method, target_idx, seed, config, vlmodel, preprocess_train, pipe):
    """
    Run a single experiment.
    
    Args:
        method: "eeg_guidance", "target_image_guidance", "random_generation"
        target_idx: Target image index
        seed: Random seed
        config: Configuration dictionary
    
    Returns:
        results: Dictionary containing various evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"Method: {method} | Target: {target_idx} | Seed: {seed}")
    print(f"{'='*80}")
    
    # Set random seeds
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Validate indices
    assert target_idx < len(image_list), f"target_idx exceeds image count ({len(image_list)})"
    assert target_idx < len(embed_list), f"target_idx exceeds embedding count ({len(embed_list)})"
    
    target_image_path = os.path.join(image_dir, image_list[target_idx])
    target_eeg_embed_path = os.path.join(embed_dir, embed_list[target_idx])
    print(f"Current index: {target_idx}")
    print(f"Image: {target_image_path}")
    print(f"eeg_embed: {target_eeg_embed_path}")
    
    # Load target features
    target_eeg_feature = torch.load(target_eeg_embed_path, weights_only=False)
    
    # Load target image CLIP embedding
    target_image_pil = Image.open(target_image_path).convert("RGB")
    with torch.no_grad():
        target_clip_embed = vlmodel.encode_image(preprocess_train(target_image_pil).unsqueeze(0).to(device))
    
    # Select optimization target based on method
    if method == "eeg_guidance":
        target_feature = target_eeg_feature  # use EEG feature
    elif method == "target_image_guidance":
        target_feature = target_clip_embed  # use CLIP embedding
    elif method == "random_generation":
        target_feature = target_eeg_feature  # not used by random method, but needed as placeholder
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Remaining parameters
    sub = 'sub-01'
    fs = 250
    selected_channel_idxes = slice(None)
    dnn = 'alexnet'
    encoding_model_path = f'/home/ldy/Workspace/Closed_loop_optimizing/kyw/closed-loop/EEG-encoding/EEG_encoder/results/{sub}/synthetic_eeg_data/encoding-end_to_end/dnn-{dnn}/modeled_time_points-all/pretrained-True/lr-1e-05__wd-0e+00__bs-064/model_state_dict.pt'
    f_encoder = "/home/ldy/Workspace/Closed_loop_optimizing/kyw/closed-loop/sub_model/sub-01/diffusion_alexnet/pretrained_True/gene_gene/ATM_S_reconstruction_scale_0_1000_40.pth"
    checkpoint = torch.load(f_encoder, map_location=device, weights_only=False)
    eeg_model = ATMS()
    eeg_model.load_state_dict(checkpoint['eeg_model_state_dict'])
    
    # Get image pool
    test_images_path = get_image_pool(image_dir)
    print(f"test_images_path {len(test_images_path)}")
    
    if target_image_path in test_images_path:
        test_images_path.remove(target_image_path)
    
    # Use externally provided test_set_img_embeds (avoid redundant loading)
    test_set_img_embeds = config['test_set_img_embeds']
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_save_dir = os.path.join(config['save_path'], timestamp, method, f'target_{target_idx}_seed_{seed}')
    os.makedirs(exp_save_dir, exist_ok=True)
    plots_save_folder = os.path.join(exp_save_dir, 'plots')
    os.makedirs(plots_save_folder, exist_ok=True)
    
    # Force GPU memory cleanup before experiment starts
    torch.cuda.empty_cache()
    
    # Initialize Generator (load IP adapter on first run, skip on subsequent experiments)
    load_ip = config.get('first_run', False)
    if load_ip:
        config['first_run'] = False  # mark as loaded
    
    # Get minimum data count threshold from config
    min_data_threshold = config.get('min_data_threshold', 10)
    Generator = HeuristicGenerator(pipe, vlmodel, preprocess_train, device=device, seed=seed, 
                                   load_ip_adapter=load_ip, min_data_threshold=min_data_threshold)
    
    # Iterative optimization
    num_loops = config['num_loops']
    processed_paths = set()
    
    all_chosen_rewards = []
    all_chosen_losses = []
    all_chosen_images = []
    all_chosen_eegs = []
    
    history_cs = []
    history_eeg = []
    fit_images = []
    fit_eegs = []
    fit_rewards = []
    fit_losses = []
    
    # Initialize last-round data (for final evaluation)
    final_loop_images = []
    final_loop_eegs = []
    final_loop_rewards = []
    final_loop_losses = []
    final_loop_eeg_scores = []  # for Top-5 selection based on EEG score
    
    for t in range(num_loops):
        print(f"Loop {t + 1}/{num_loops}")
        loop_save_dir = os.path.join(exp_save_dir, f'loop_{t+1}')
        os.makedirs(loop_save_dir, exist_ok=True)
        loop_sample_ten = []
        loop_reward_ten = []
        loop_eeg_ten = []
        loop_loss_ten = []
        
        if method == "random_generation":
            # Method 3: Random Generation (Null baseline) - generate random images
            print("Method 3: Random Generation (Null baseline) - Generating random images...")
            
            # Only execute in the first round; subsequent rounds are not needed
            if t == 0:
                # Create temporary generator for random generation (skip IP adapter to save memory)
                print("Creating temporary generator for random image generation...")
                temp_generator = torch.Generator(device=device).manual_seed(seed)
                
                # Generate 5 random images
                NUM_RANDOM_IMAGES = 5
                
                # Create random noise
                epsilon = torch.randn(
                    Generator.num_inference_steps + 1, 
                    NUM_RANDOM_IMAGES,
                    Generator.pipe.unet.config.in_channels,
                    Generator.pipe.unet.config.sample_size,
                    Generator.pipe.unet.config.sample_size,
                    device=device, 
                    generator=temp_generator
                )
                
                # Randomly initialize pseudo target (no optimization)
                random_pseudo_target = torch.randn(
                    NUM_RANDOM_IMAGES, 
                    Generator.dimension, 
                    device=device, 
                    generator=temp_generator
                )
                
                print(f"Generating {NUM_RANDOM_IMAGES} random images without optimization...")
                # Generate directly without optimization loop
                latents = Generator.pipe(
                    [''] * NUM_RANDOM_IMAGES,
                    ip_adapter_image_embeds=[random_pseudo_target.unsqueeze(0).type(torch.bfloat16).to(device)],
                    latents=epsilon[0].type(torch.bfloat16),
                    given_noise=epsilon[1:].type(torch.bfloat16),
                    output_type="latent",
                    num_inference_steps=Generator.num_inference_steps,
                    guidance_scale=Generator.guidance_scale,
                    eta=1.0,
                ).images
                
                # Decode one image at a time to avoid OOM
                chosen_images = []
                for i in range(NUM_RANDOM_IMAGES):
                    img = Generator.latents_to_images(latents[i:i+1])  # decode 1 image at a time
                    chosen_images.extend(img)
                    del img
                    if i % 2 == 0:  # clean up every 2 images
                        torch.cuda.empty_cache()
                
                # Compute EEG (for calculating EEG score)
                synthetic_eegs = generate_eeg_from_image(encoding_model_path, chosen_images, device)
                chosen_eegs = synthetic_eegs
                
                # Compute rewards (only for interface consistency; random baseline does not need rewards)
                chosen_rewards = []
                for idx, eeg in enumerate(synthetic_eegs):
                    # Record actual EEG similarity (not used for optimization)
                    cs, _ = reward_function_clip_embed(eeg, eeg_model, target_eeg_feature, sub, device)
                    chosen_rewards.append(cs)
                
                chosen_losses = [0] * len(chosen_images)
                
                loop_sample_ten.extend(chosen_images)
                loop_eeg_ten.extend(chosen_eegs)
                loop_reward_ten.extend(chosen_rewards)
                loop_loss_ten.extend(chosen_losses)
                
                # Clean up
                del epsilon, random_pseudo_target, latents, synthetic_eegs, temp_generator
                torch.cuda.empty_cache()
                
                print(f"Random baseline rewards: {[f'{r:.4f}' for r in chosen_rewards]}")
                print(f"Random baseline mean reward: {np.mean(chosen_rewards):.4f}")
                
                # Save data for final evaluation (random baseline)
                final_loop_images = loop_sample_ten.copy()
                final_loop_eegs = loop_eeg_ten.copy()
                final_loop_rewards = loop_reward_ten.copy()
                final_loop_losses = loop_loss_ten.copy()
                final_loop_eeg_scores = chosen_rewards.copy()  # Random rewards are the EEG scores
                
                # Random baseline completes after the first round, break directly
                # No need for multiple iterations
                print("Random generation baseline completed (single round generation)")
            else:
                # Should not reach here since random_generation only needs the first round
                pass
        
        else:
            # Method 1 & 2: Normal optimization flow
            if t == 0:
                chosen_rewards, chosen_losses, chosen_images, chosen_eegs = get_prob_random_sample(
                    test_images_path, encoding_model_path, fs, device, selected_channel_idxes,
                    processed_paths, target_feature, size=4, method=method, 
                    eeg_model=eeg_model, sub=sub, dnn=dnn, 
                    vlmodel=vlmodel, preprocess_train=preprocess_train)
                
                loop_sample_ten.extend(chosen_images)
                loop_eeg_ten.extend(chosen_eegs)
                loop_reward_ten.extend(chosen_rewards)
                loop_loss_ten.extend(chosen_losses)
                
                tensor_loop_sample_ten = [preprocess_train(i) for i in loop_sample_ten]    
                with torch.no_grad():
                    tensor_loop_sample_ten_embeds = vlmodel.encode_image(torch.stack(tensor_loop_sample_ten).to(device))        
                Generator.pseudo_target_model.add_model_data(
                    tensor_loop_sample_ten_embeds.clone(),
                    (-torch.tensor(loop_reward_ten) * Generator.reward_scaling_factor).to(device))
                
                # Clean up temporary variables
                del tensor_loop_sample_ten, tensor_loop_sample_ten_embeds
                torch.cuda.empty_cache()
            
            else:
                # Image fusion generation (multiple fusions per round)
                tensor_fit_images = [preprocess_train(i) for i in fit_images]    
                with torch.no_grad():
                    img_embeds = vlmodel.encode_image(torch.stack(tensor_fit_images).to(device))    
                
                # Clean up temporary tensors
                del tensor_fit_images
                torch.cuda.empty_cache()
                
                # Key change: perform multiple fusions per round
                NUM_FUSIONS_PER_ROUND = 2  # increased from 1 to 2
                print(f"[INFO] Performing {NUM_FUSIONS_PER_ROUND} fusions in this round")
                
                all_generated_images = []
                all_fusion_source_images = []
                
                for fusion_idx in range(NUM_FUSIONS_PER_ROUND):
                    print(f"  Fusion {fusion_idx + 1}/{NUM_FUSIONS_PER_ROUND}...")
                    
                    # Fusion generation, also return indices of original images used for fusion
                    generated_images, (idx1, idx2) = fusion_image_to_images(Generator, img_embeds, fit_rewards, device, 512, save_path=loop_save_dir)
                    synthetic_eegs = generate_eeg_from_image(encoding_model_path, generated_images, device)
                    
                    # Add fusion-generated images
                    loop_sample_ten.extend(generated_images)
                    loop_eeg_ten.extend(synthetic_eegs)
                    all_generated_images.extend(generated_images)
                    
                    # Compute rewards for fusion-generated images
                    for idx, eeg in enumerate(synthetic_eegs):
                        if method == "eeg_guidance":
                            cs, eeg_feature = reward_function_clip_embed(eeg, eeg_model, target_feature, sub, device)
                        elif method == "target_image_guidance":
                            cs = reward_function_clip_embed_image(generated_images[idx], target_feature, vlmodel, preprocess_train, device)
                        loss = 0
                        loop_reward_ten.append(cs)
                        loop_loss_ten.append(loss)
                    
                    # Clean up EEG data
                    del synthetic_eegs
                    torch.cuda.empty_cache()
                    
                    # Add original images used for fusion
                    print(f"  Adding fusion source images (idx {idx1} and {idx2})")
                    fusion_source_images = [fit_images[idx1], fit_images[idx2]]
                    fusion_source_eegs = generate_eeg_from_image(encoding_model_path, fusion_source_images, device)
                    
                    loop_sample_ten.extend(fusion_source_images)
                    loop_eeg_ten.extend(fusion_source_eegs)
                    all_fusion_source_images.extend(fusion_source_images)
                    
                    # Compute rewards for fusion source images
                    for idx, eeg in enumerate(fusion_source_eegs):
                        if method == "eeg_guidance":
                            cs, eeg_feature = reward_function_clip_embed(eeg, eeg_model, target_feature, sub, device)
                        elif method == "target_image_guidance":
                            cs = reward_function_clip_embed_image(fusion_source_images[idx], target_feature, vlmodel, preprocess_train, device)
                        loss = 0
                        loop_reward_ten.append(cs)
                        loop_loss_ten.append(loss)
                    
                    # Clean up
                    del fusion_source_eegs
                    torch.cuda.empty_cache()
                
                print(f"[INFO] Fusion complete: generated {len(all_generated_images)} new images, added {len(all_fusion_source_images)} source images")
                
                # Greedy sampling (improved): sample based on target instead of generated images
                # Key change: use target_clip_embed instead of img_embeds
                greedy_images = []
                TOP_K = 20  # increase top-k range for more diversity
                # Match greedy sample count with fusion generation count
                NUM_GREEDY_SAMPLES = len(all_generated_images)  # total images from fusion per round
                
                # Get indices of available images
                available_indices = []
                for i, path in enumerate(test_images_path):
                    if path not in processed_paths:
                        available_indices.append(i)
                
                if len(available_indices) > 0:
                    # Key change: compute similarity based on target_clip_embed
                    available_features = test_set_img_embeds[available_indices]
                    cosine_similarities = compute_embed_similarity(
                        target_clip_embed.to(device), 
                        available_features.to(device)
                    )
                    sorted_available_indices = np.argsort(cosine_similarities.cpu())
                    
                    # Randomly sample multiple images from top-K
                    top_indices = sorted_available_indices[-min(TOP_K, len(sorted_available_indices)):]
                    
                    # Randomly select NUM_GREEDY_SAMPLES images from top-K (without replacement)
                    num_to_sample = min(NUM_GREEDY_SAMPLES, len(top_indices))
                    selected_indices = np.random.choice(top_indices, size=num_to_sample, replace=False)
                    
                    sample_image_paths = []
                    for selected_idx in selected_indices:
                        greedy_image = Image.open(test_images_path[available_indices[selected_idx]]).convert("RGB")
                        greedy_images.append(greedy_image)
                        sample_image_paths.append(test_images_path[available_indices[selected_idx]])
                    
                    processed_paths.update(sample_image_paths)
                    print(f"[INFO] Greedy sampling: selected {len(greedy_images)} images from top-{TOP_K} similar to target")
                
                synthetic_eegs = generate_eeg_from_image(encoding_model_path, greedy_images, device)
                loop_sample_ten.extend(greedy_images)
                loop_eeg_ten.extend(synthetic_eegs)
                
                for idx, eeg in enumerate(synthetic_eegs):
                    if method == "eeg_guidance":
                        cs, eeg_feature = reward_function_clip_embed(eeg, eeg_model, target_feature, sub, device)
                    elif method == "target_image_guidance":
                        cs = reward_function_clip_embed_image(greedy_images[idx], target_feature, vlmodel, preprocess_train, device)
                    loss = 0
                    loop_reward_ten.append(cs)
                    loop_loss_ten.append(loss)
                
                # Clean up EEG and img_embeds
                del synthetic_eegs, img_embeds
                torch.cuda.empty_cache()
                
                # Select top-4
                loop_probabilities = softmax(loop_reward_ten)
                chosen_rewards, chosen_losses, chosen_images, chosen_eegs = select_from_images(
                    loop_probabilities, loop_reward_ten, loop_loss_ten, loop_sample_ten, loop_eeg_ten, size=4)
                
                # Sort by reward
                combined = list(zip(chosen_rewards, chosen_losses, chosen_images, chosen_eegs))
                combined.sort(reverse=True, key=lambda x: x[0])
                chosen_rewards, chosen_losses, chosen_images, chosen_eegs = zip(*combined)
                chosen_rewards = list(chosen_rewards)
                chosen_losses = list(chosen_losses)
                chosen_images = list(chosen_images)
                chosen_eegs = list(chosen_eegs)
                
                tensor_loop_sample_ten = [preprocess_train(i) for i in loop_sample_ten]    
                with torch.no_grad():
                    tensor_loop_sample_ten_embeds = vlmodel.encode_image(torch.stack(tensor_loop_sample_ten).to(device))        
                Generator.pseudo_target_model.add_model_data(
                    tensor_loop_sample_ten_embeds.clone(),
                    (-torch.tensor(loop_reward_ten) * Generator.reward_scaling_factor).to(device))
                
                # Clean up temporary variables (Method 1 & 2 branch)
                del tensor_loop_sample_ten, tensor_loop_sample_ten_embeds
                torch.cuda.empty_cache()
        
        # Update fit data
        fit_images = chosen_images
        fit_eegs = chosen_eegs
        fit_rewards = chosen_rewards
        fit_losses = chosen_losses
        
        all_chosen_rewards.extend(chosen_rewards)
        all_chosen_losses.extend(chosen_losses)
        all_chosen_images.extend(chosen_images)
        all_chosen_eegs.extend(chosen_eegs)
        
        # Visualize
        visualize_top_images(loop_sample_ten, loop_reward_ten, loop_save_dir, t)
        
        # Record historical best
        max_similarity = max(loop_reward_ten)
        max_index = loop_reward_ten.index(max_similarity)
        corresponding_eeg = loop_eeg_ten[max_index]
        
        if len(history_cs) == 0:
            history_cs.append(max_similarity)
            history_eeg.append(corresponding_eeg)
        else:
            max_history = max(history_cs)
            if max_similarity > max_history:
                history_cs.append(max_similarity)
                history_eeg.append(corresponding_eeg)
            else:
                history_cs.append(max_history)
                history_eeg.append(history_eeg[-1])
        
        # Random baseline only needs one round, end directly
        if method == "random_generation" and t == 0:
            print("Random baseline completed after first round.")
            break
        
        # Early stopping check
        if len(history_cs) >= 2:
            if history_cs[-1] != history_cs[-2]:
                diff = abs(history_cs[-1] - history_cs[-2])
                print(f"History: {history_cs[-1]:.4f}, {history_cs[-2]:.4f}, diff={diff:.6f}")
                if diff <= 1e-4:
                    print("The difference is within 1e-4, stopping early.")
                    break
        
        # Save last round data for final evaluation
        if t == num_loops - 1:  # last round
            final_loop_images = loop_sample_ten.copy()
            final_loop_eegs = loop_eeg_ten.copy()
            final_loop_rewards = loop_reward_ten.copy()
            final_loop_losses = loop_loss_ten.copy()
            
            # Compute actual EEG scores for final selection (regardless of optimization target)
            print(f"\n[INFO] Computing EEG scores for final selection...")
            final_loop_eeg_scores = []
            for eeg in loop_eeg_ten:
                eeg_score, _ = reward_function_clip_embed(eeg, eeg_model, target_eeg_feature, sub, device)
                final_loop_eeg_scores.append(eeg_score)
            print(f"Last round EEG scores: {[f'{s:.4f}' for s in final_loop_eeg_scores]}")
        
        # Clean up per-round temporary data
        del loop_sample_ten, loop_eeg_ten, loop_reward_ten, loop_loss_ten
        torch.cuda.empty_cache()
    
    # Plot similarity curve
    plt.figure(figsize=(10, 5))
    plt.plot(history_cs, marker='o', markersize=3, label='Similarity')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend()
    path = os.path.join(exp_save_dir, 'similarities.jpg')
    plt.savefig(path)
    plt.close()
    
    print(f"\nFinal Statistics:")
    print(f"all_chosen_images: {len(all_chosen_images)}")
    print(f"all_chosen_eegs: {len(all_chosen_eegs)}")
    print(f"history_cs: {history_cs}")
    
    # === Evaluation phase ===
    # Final evaluation: select Top-5 based on EEG score (deterministic)
    # Rationale: regardless of optimization strategy, the ultimate goal is to produce the desired EEG features
    if method == "random_generation":
        # Random baseline uses 5 images
        final_images = chosen_images  # should be 5 images
        final_chosen_eegs = chosen_eegs  # corresponding EEGs
        print(f"\nRandom baseline: evaluating all {len(final_images)} images (no selection)")
    else:
        # Other methods: select Top-5 based on EEG score from all candidates (8 images) in the last round
        if len(final_loop_images) > 0:
            print(f"\n[INFO] Final evaluation: Selecting Top-5 based on EEG scores from last round's {len(final_loop_images)} candidates")
            print(f"Last round optimization rewards ({method}): {[f'{r:.4f}' for r in final_loop_rewards]}")
            
            # Key: select Top-5 based on EEG score (not optimization reward)
            # This ensures evaluation of images truly optimal for EEG features
            final_chosen_eeg_scores, final_chosen_losses, final_images, final_chosen_eegs = select_top_k_images(
                final_loop_eeg_scores, final_loop_losses, final_loop_images, final_loop_eegs, size=5)
            
            print(f"Selected Top-5 EEG scores: {[f'{s:.4f}' for s in final_chosen_eeg_scores]}")
            print(f"{method}: evaluating {len(final_images)} images (Top-5 by EEG score)")
        else:
            # If stopped early, use the last selected images
            print(f"[WARNING] No final loop data available, using last chosen images")
            final_images = chosen_images
            final_chosen_eegs = chosen_eegs
            print(f"{method}: evaluating {len(final_images)} images")
    
    # Save final images
    saved_image_paths = []
    for i, img in enumerate(final_images):
        save_name = os.path.join(exp_save_dir, f"final_generated_{i}.png")
        img.save(save_name)
        saved_image_paths.append(save_name)
    
    # Compute evaluation metrics
    # Unified strategy: all methods report EEG score (primary) and CLIP score (secondary)
    # Rationale: the task objective is to optimize visual stimuli to produce desired EEG features
    
    print("\n" + "="*80)
    print("FINAL EVALUATION METRICS")
    print("="*80)
    
    # 1. Compute EEG scores (Primary Metric - the true task objective)
    print("\n[Primary Metric] Computing EEG similarity...")
    if method != "random_generation" and len(final_loop_images) > 0:
        # Optimization methods: already have final_chosen_eegs, use directly
        eeg_scores = []
        for eeg in final_chosen_eegs:
            score, _ = reward_function_clip_embed(eeg, eeg_model, target_eeg_feature, sub, device)
            eeg_scores.append(score)
    else:
        # Random baseline or early stop: need to recompute
        final_eegs = generate_eeg_from_image(encoding_model_path, final_images, device)
        eeg_scores = []
        for final_eeg in final_eegs:
            score, _ = reward_function_clip_embed(final_eeg, eeg_model, target_eeg_feature, sub, device)
            eeg_scores.append(score)
        del final_eegs
        torch.cuda.empty_cache()
    
    print(f"EEG Scores: {[f'{s:.4f}' for s in eeg_scores]}")
    print(f"Average EEG Score: {np.mean(eeg_scores):.4f} ± {np.std(eeg_scores):.4f}")
    print(f"Max EEG Score: {np.max(eeg_scores):.4f}")
    print(f"Min EEG Score: {np.min(eeg_scores):.4f}")
    
    # 2. Compute CLIP scores (Secondary Metric - reference metric)
    print("\n[Secondary Metric] Computing CLIP similarity...")
    clip_scores = []
    for img in final_images:
        score = reward_function_clip_embed_image(img, target_clip_embed, vlmodel, preprocess_train, device)
        clip_scores.append(score)
    
    print(f"CLIP Scores: {[f'{s:.4f}' for s in clip_scores]}")
    print(f"Average CLIP Score: {np.mean(clip_scores):.4f} ± {np.std(clip_scores):.4f}")
    print(f"Max CLIP Score: {np.max(clip_scores):.4f}")
    print(f"Min CLIP Score: {np.min(clip_scores):.4f}")
    
    print("="*80)
    
    # Thoroughly clean up all large objects
    # First clean up models inside Generator
    if hasattr(Generator, 'pseudo_target_model'):
        del Generator.pseudo_target_model
    del Generator
    
    del eeg_model
    del target_eeg_feature
    del target_clip_embed
    del final_images
    del chosen_images, chosen_eegs, chosen_rewards, chosen_losses
    del fit_images, fit_eegs, fit_rewards, fit_losses
    del all_chosen_images, all_chosen_eegs, all_chosen_rewards, all_chosen_losses
    del history_eeg
    
    # Force GPU memory and RAM cleanup
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Return results
    results = {
        'method': method,
        'target_idx': int(target_idx),
        'seed': int(seed),
        'eeg_scores': eeg_scores,
        'clip_scores': clip_scores,
        'avg_eeg_score': float(np.mean(eeg_scores)),
        'avg_clip_score': float(np.mean(clip_scores)),
        'saved_paths': saved_image_paths,
        'history_rewards': [float(x) for x in history_cs]
    }
    
    print(f"Results: EEG={results['avg_eeg_score']:.4f}, CLIP={results['avg_clip_score']:.4f}")
    return results


def generate_benchmark_report(all_results, config):
    """Generate benchmark report.
    
    Note:
    - random_generation uses 5 randomly sampled images (true unbiased baseline)
    - eeg_guidance and target_image_guidance use 4 optimized images
    """
    save_path = config['save_path']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Create DataFrame
    records = []
    for result in all_results:
        for i, (eeg_score, clip_score) in enumerate(zip(result['eeg_scores'], result['clip_scores'])):
            records.append({
                'Method': result['method'],
                'Target_Idx': result['target_idx'],
                'Seed': result['seed'],
                'Image_Idx': i,
                'EEG_Score': eeg_score,
                'CLIP_Score': clip_score,
                'Num_Images': len(result['eeg_scores'])  # record image count
            })
    
    df = pd.DataFrame(records)
    
    # Save detailed results
    csv_path = os.path.join(save_path, f'detailed_results_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")
    
    # 2. Compute summary statistics
    summary_stats = df.groupby('Method').agg({
        'EEG_Score': ['mean', 'std', 'min', 'max'],
        'CLIP_Score': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    summary_path = os.path.join(save_path, f'summary_statistics_{timestamp}.csv')
    summary_stats.to_csv(summary_path)
    print(f"Summary statistics saved to: {summary_path}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY STATISTICS")
    print(f"{'='*80}")
    print("\nNote:")
    print("  - random_generation: 5 randomly generated images (unoptimized null baseline)")
    print("  - eeg_guidance: 4 images optimized with EEG features, primary metric is EEG Score")
    print("  - target_image_guidance: 4 images optimized with target image CLIP features, primary metric is CLIP Score")
    print("\nEvaluation description:")
    print("  - EEG Score: generated image EEG features vs target image EEG features (eeg_guidance optimization target)")
    print("  - CLIP Score: generated image CLIP embedding vs target image CLIP embedding (target_image_guidance optimization target)")
    print("  - random_generation: images generated from random noise without any optimization (consistent with Offline method baseline)")
    print(f"\n{'='*80}")
    print(summary_stats)
    print(f"{'='*80}\n")
    
    # Count images per method
    print("Image count statistics per method:")
    for method in df['Method'].unique():
        method_df = df[df['Method'] == method]
        num_images_per_exp = method_df.groupby(['Target_Idx', 'Seed']).size().unique()
        print(f"  {method}: {num_images_per_exp[0] if len(num_images_per_exp) == 1 else num_images_per_exp} images/experiment")
    print()
    
    # 3. Generate visual comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # EEG Score comparison
    method_names = {
        'eeg_guidance': 'EEG Feature\nGuidance',
        'target_image_guidance': 'Target Image\nCLIP Guidance',
        'random_generation': 'Random\nGeneration'
    }
    df['Method_Label'] = df['Method'].map(method_names)
    
    sns.boxplot(data=df, x='Method_Label', y='EEG_Score', ax=axes[0])
    axes[0].set_title('EEG Similarity Score', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Method', fontsize=12)
    axes[0].set_ylabel('EEG Score', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # CLIP Score comparison
    sns.boxplot(data=df, x='Method_Label', y='CLIP_Score', ax=axes[1])
    axes[1].set_title('CLIP Similarity Score', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Method', fontsize=12)
    axes[1].set_ylabel('CLIP Score', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, f'benchmark_comparison_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    plt.close()
    
    # 4. Generate bar chart comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    summary_mean = df.groupby('Method')[['EEG_Score', 'CLIP_Score']].mean()
    summary_std = df.groupby('Method')[['EEG_Score', 'CLIP_Score']].std()
    
    methods = list(method_names.keys())
    method_labels = [method_names[m] for m in methods]
    
    # EEG Score bar chart
    eeg_means = [summary_mean.loc[m, 'EEG_Score'] if m in summary_mean.index else 0 for m in methods]
    eeg_stds = [summary_std.loc[m, 'EEG_Score'] if m in summary_std.index else 0 for m in methods]
    axes[0].bar(method_labels, eeg_means, yerr=eeg_stds, capsize=5, alpha=0.7, color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[0].set_title('EEG Similarity (Mean ± Std)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('EEG Score', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # CLIP Score bar chart
    clip_means = [summary_mean.loc[m, 'CLIP_Score'] if m in summary_mean.index else 0 for m in methods]
    clip_stds = [summary_std.loc[m, 'CLIP_Score'] if m in summary_std.index else 0 for m in methods]
    axes[1].bar(method_labels, clip_means, yerr=clip_stds, capsize=5, alpha=0.7, color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[1].set_title('CLIP Similarity (Mean ± Std)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('CLIP Score', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    bar_plot_path = os.path.join(save_path, f'benchmark_barplot_{timestamp}.png')
    plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
    print(f"Bar chart saved to: {bar_plot_path}")
    plt.close()
    
    # 5. Save complete results in JSON format
    summary_dict = {}
    for method in summary_stats.index:
        summary_dict[method] = {
            'EEG_Score': {
                'mean': float(summary_stats.loc[method, ('EEG_Score', 'mean')]),
                'std': float(summary_stats.loc[method, ('EEG_Score', 'std')]),
                'min': float(summary_stats.loc[method, ('EEG_Score', 'min')]),
                'max': float(summary_stats.loc[method, ('EEG_Score', 'max')])
            },
            'CLIP_Score': {
                'mean': float(summary_stats.loc[method, ('CLIP_Score', 'mean')]),
                'std': float(summary_stats.loc[method, ('CLIP_Score', 'std')]),
                'min': float(summary_stats.loc[method, ('CLIP_Score', 'min')]),
                'max': float(summary_stats.loc[method, ('CLIP_Score', 'max')])
            }
        }
    
    # Create serializable config copy (excluding Tensor objects)
    config_serializable = {
        'target_indices': config['target_indices'],
        'num_seeds': config['num_seeds'],
        'num_loops': config['num_loops'],
        'save_path': config['save_path'],
        'methods': config['methods']
    }
    
    json_results = {
        'config': config_serializable,
        'timestamp': timestamp,
        'summary_statistics': summary_dict,
        'all_results': all_results
    }
    
    json_path = os.path.join(save_path, f'complete_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Complete results JSON saved to: {json_path}")
    
    # 6. Generate dedicated Chance Level analysis report
    if 'random_generation' in df['Method'].unique():
        random_df = df[df['Method'] == 'random_generation']
        
        chance_level_report = []
        chance_level_report.append("=" * 80)
        chance_level_report.append("CHANCE LEVEL (Random Baseline) Detailed Analysis")
        chance_level_report.append("=" * 80)
        chance_level_report.append("")
        chance_level_report.append("Method description:")
        chance_level_report.append("  Generates 5 images from random noise (without considering any target features)")
        chance_level_report.append("  Does not include any optimization or guidance process")
        chance_level_report.append("  Uses randomly initialized pseudo target embedding")
        chance_level_report.append("  Consistent with the Offline Generation random baseline implementation")
        chance_level_report.append("")
        chance_level_report.append("Statistical results:")
        chance_level_report.append(f"  EEG Score:")
        chance_level_report.append(f"    Mean: {random_df['EEG_Score'].mean():.4f}")
        chance_level_report.append(f"    Std dev: {random_df['EEG_Score'].std():.4f}")
        chance_level_report.append(f"    Min: {random_df['EEG_Score'].min():.4f}")
        chance_level_report.append(f"    Max: {random_df['EEG_Score'].max():.4f}")
        chance_level_report.append(f"    Median: {random_df['EEG_Score'].median():.4f}")
        chance_level_report.append("")
        chance_level_report.append(f"  CLIP Score:")
        chance_level_report.append(f"    Mean: {random_df['CLIP_Score'].mean():.4f}")
        chance_level_report.append(f"    Std dev: {random_df['CLIP_Score'].std():.4f}")
        chance_level_report.append(f"    Min: {random_df['CLIP_Score'].min():.4f}")
        chance_level_report.append(f"    Max: {random_df['CLIP_Score'].max():.4f}")
        chance_level_report.append(f"    Median: {random_df['CLIP_Score'].median():.4f}")
        chance_level_report.append("")
        chance_level_report.append(f"  Sample count: {len(random_df)} images")
        chance_level_report.append(f"  Experiment count: {random_df.groupby(['Target_Idx', 'Seed']).ngroups} runs")
        chance_level_report.append("")
        chance_level_report.append("Comparison with other methods:")
        chance_level_report.append("  Note: different methods have different optimization targets")
        chance_level_report.append("  - eeg_guidance: optimization target is EEG Score (primary metric)")
        chance_level_report.append("  - target_image_guidance: optimization target is CLIP Score (primary metric)")
        chance_level_report.append("")
        
        for method in ['eeg_guidance', 'target_image_guidance']:
            if method in df['Method'].unique():
                method_df = df[df['Method'] == method]
                eeg_improvement = (method_df['EEG_Score'].mean() - random_df['EEG_Score'].mean()) / random_df['EEG_Score'].mean() * 100
                clip_improvement = (method_df['CLIP_Score'].mean() - random_df['CLIP_Score'].mean()) / random_df['CLIP_Score'].mean() * 100
                
                if method == 'eeg_guidance':
                    chance_level_report.append(f"  {method}:")
                    chance_level_report.append(f"    EEG Score improvement: {eeg_improvement:+.2f}% <- primary optimization target")
                    chance_level_report.append(f"    CLIP Score improvement: {clip_improvement:+.2f}%")
                else:  # target_image_guidance
                    chance_level_report.append(f"  {method}:")
                    chance_level_report.append(f"    EEG Score improvement: {eeg_improvement:+.2f}%")
                    chance_level_report.append(f"    CLIP Score improvement: {clip_improvement:+.2f}% <- primary optimization target")
                chance_level_report.append("")
        
        chance_level_report.append("=" * 80)
        
        chance_level_text = "\n".join(chance_level_report)
        print("\n" + chance_level_text)
        
        # Save to file
        chance_level_path = os.path.join(save_path, f'chance_level_analysis_{timestamp}.txt')
        with open(chance_level_path, 'w') as f:
            f.write(chance_level_text)
        print(f"\nChance Level analysis saved to: {chance_level_path}")
    
    return df, summary_stats


def main():
    """Main function: run benchmark for three methods"""
    
    # Initialize shared models
    print("Loading shared models...")
    model_type = 'ViT-H-14'
    vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
        model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device=device)
    vlmodel.to(device)
    
    generator = Generator4Embeds(device=device)
    pipe = generator.pipe
    
    # Load test set image embeddings (load once, shared across all experiments)
    print("Loading test set image embeddings...")
    test_set_img_embeds = torch.load("/mnt/dataset1/ldy/Workspace/FLORA/data_preparing/ViT-H-14_features_test.pt")['img_features'].cpu()
    print(f"Loaded {test_set_img_embeds.shape[0]} image embeddings")
    
    # Configure parameters
    config = {
        # 'target_indices': [i for i in range(30)],  # target image indices to test
        'target_indices': np.linspace(1, 200, 10, dtype=int).tolist(),
        'num_seeds': 1,  # number of random seeds per method
        'num_loops': 10,  # number of iteration rounds per experiment
        'save_path': '/home/ldy/Workspace/Closed_loop_optimizing/outputs/benchmark_heuristic_generation',
        'methods': ['eeg_guidance', 'target_image_guidance', 'random_generation'],
        'test_set_img_embeds': test_set_img_embeds,  # shared image embeddings
        'first_run': True,  # flag for first run, needs to load IP adapter
        'min_data_threshold': 10  # minimum data count threshold; optimization is skipped below this value
    }
    
    os.makedirs(config['save_path'], exist_ok=True)
    
    print(f"\n{'='*80}")
    print("HEURISTIC GENERATION BENCHMARK")
    print(f"{'='*80}")
    print(f"Methods: {config['methods']}")
    print(f"Target indices: {config['target_indices']}")
    print(f"Seeds per method: {config['num_seeds']}")
    print(f"Num loops per experiment: {config['num_loops']}")
    print(f"Min data threshold: {config['min_data_threshold']} (optimization starts when data >= this value)")
    print(f"{'='*80}\n")
    
    all_results = []
    
    # Run all experiments
    for target_idx in config['target_indices']:
        for method in config['methods']:
            for seed in range(config['num_seeds']):
                try:
                    # Force cleanup before experiment
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    result = run_single_experiment(method, target_idx, seed, config, vlmodel, preprocess_train, pipe)
                    all_results.append(result)
                    
                    # Force cleanup after experiment
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"\n!!! Error in {method}, target {target_idx}, seed {seed}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Force cleanup after error
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
    
    # Clean up shared test_set_img_embeds
    if 'test_set_img_embeds' in config:
        del config['test_set_img_embeds']
    torch.cuda.empty_cache()
    
    # Generate report
    if all_results:
        print(f"\n{'='*80}")
        print("GENERATING BENCHMARK REPORT...")
        print(f"{'='*80}\n")
        df, summary_stats = generate_benchmark_report(all_results, config)
        
        print(f"\n{'='*80}")
        print("BENCHMARK COMPLETED!")
        print(f"{'='*80}")
        print(f"Total experiments run: {len(all_results)}")
        print(f"Results saved to: {config['save_path']}")
        print(f"{'='*80}\n")
    else:
        print("\n!!! No results to report. All experiments failed.")
    
    # Final cleanup
    del vlmodel, pipe, generator
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

