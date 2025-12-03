"""
Benchmark for Heuristic Generation with Three Different Methods:
1. EEG Feature Guidance - 使用目标脑电特征作为优化目标
2. Target Image CLIP Guidance - 使用目标图像的CLIP embedding作为优化目标
3. Random Generation - 完全随机生成，不做任何优化（baseline）

评估方式：
所有方法都使用相同的评估标准：
- EEG Score: 生成图像 vs 目标图像的脑电特征相似度
- CLIP Score: 生成图像 vs 目标图像的CLIP embedding相似度
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

# 配置图片和embed目录
image_dir = '/home/ldy/Workspace/Closed_loop_optimizing/test_images'
embed_dir = '/home/ldy/Workspace/Closed_loop_optimizing/data/clip_embed/2025-09-21'

# 获取所有图片和embed路径，排序保证一一对应
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
        self.min_data_threshold = min_data_threshold  # 最小数据量阈值

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

        # Optimization loop - 只优化 pseudo_target，不生成图像
        for step in range(self.total_steps):
            data_x, data_y = self.pseudo_target_model.get_model_data()   
            if data_y.size(0) < self.min_data_threshold:  # 检查数据量是否足够
                print(f"[WARNING] Insufficient data ({data_y.size(0)} < {self.min_data_threshold}), returning random generation")
                # 数据不足，生成随机图像并返回
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
            
            # 数据足够，只进行 pseudo_target 优化（不生成图像）
            step_size = self.initial_step_size / (1 + self.decay_rate * step)
            pseudo_target, _ = self.pseudo_target_model.estimate_pseudo_target(pseudo_target, step_size=step_size)

        # 优化循环结束后，用优化好的 pseudo_target 生成最终图像
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
        
        # 清理generate函数的所有中间变量
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
    """计算图像CLIP embedding与目标feature的相似度"""
    tensor_images = [preprocess_train(pil_image)]    
    with torch.no_grad():
        img_embeds = vlmodel.encode_image(torch.stack(tensor_images).to(device))      
    similarity = torch.nn.functional.cosine_similarity(img_embeds.to(device), target_feature.to(device))
    similarity = (similarity + 1) / 2    
    return similarity.item()


def reward_function_clip_embed(eeg, eeg_model, target_feature, sub, device):
    """计算EEG feature与目标feature的相似度"""
    eeg_feature = get_eeg_features(eeg_model, torch.tensor(eeg).unsqueeze(0), device, sub)    
    similarity = torch.nn.functional.cosine_similarity(eeg_feature.to(device), target_feature.to(device))
    similarity = (similarity + 1) / 2
    return similarity.item(), eeg_feature


def fusion_image_to_images(Generator, img_embeds, rewards, device, scale, save_path=None):
    """图像融合生成"""
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
    
    # 返回生成的图像和被用于融合的原始图像的索引
    return generated_images, (idx1, idx2)


def select_from_image_paths(probabilities, similarities, losses, sample_image_paths, synthetic_eegs, device, size):
    # 优化过程中使用概率采样，保持探索性和多样性
    chosen_indices = np.random.choice(len(probabilities), size=size, replace=False, p=probabilities)
    chosen_similarities = [similarities[idx] for idx in chosen_indices.tolist()] 
    chosen_losses = [losses[idx] for idx in chosen_indices.tolist()]    
    chosen_images = [Image.open(sample_image_paths[i]).convert("RGB") for i in chosen_indices.tolist()]        
    chosen_eegs = [synthetic_eegs[idx] for idx in chosen_indices.tolist()]
    return chosen_similarities, chosen_losses, chosen_images, chosen_eegs


def select_from_images(probabilities, similarities, losses, images_list, eeg_list, size):
    # 优化过程中使用概率采样，保持探索性和多样性
    chosen_indices = np.random.choice(len(similarities), size=size, replace=False, p=probabilities)
    chosen_similarities = [similarities[idx] for idx in chosen_indices.tolist()] 
    chosen_losses = [losses[idx] for idx in chosen_indices.tolist()]
    chosen_images = [images_list[idx] for idx in chosen_indices.tolist()]
    chosen_eegs = [eeg_list[idx] for idx in chosen_indices.tolist()]
    return chosen_similarities, chosen_losses, chosen_images, chosen_eegs


def select_top_k_images(similarities, losses, images_list, eeg_list, size):
    """
    选择Top-k（reward最高的k张图像）
    用于最终评估，确保选中最优的图像
    """
    sorted_indices = np.argsort(similarities)[::-1]  # 按reward降序排序
    chosen_indices = sorted_indices[:size]  # 选择前k个
    
    chosen_similarities = [similarities[idx] for idx in chosen_indices.tolist()] 
    chosen_losses = [losses[idx] for idx in chosen_indices.tolist()]
    chosen_images = [images_list[idx] for idx in chosen_indices.tolist()]
    chosen_eegs = [eeg_list[idx] for idx in chosen_indices.tolist()]
    return chosen_similarities, chosen_losses, chosen_images, chosen_eegs


def get_prob_random_sample(test_images_path, model_path, fs, device, selected_channel_idxes, 
                           processed_paths, target_feature, size, method, eeg_model, sub, dnn,
                           vlmodel, preprocess_train):
    """根据method采样图像并计算rewards"""
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
            # Method 1: 使用EEG feature相似度
            cs, eeg_feature = reward_function_clip_embed(eeg, eeg_model, target_feature, sub, device)
        elif method == "target_image_guidance":
            # Method 2: 使用CLIP embedding相似度
            cs = reward_function_clip_embed_image(pil_images[idx], target_feature, vlmodel, preprocess_train, device)
        elif method == "random_generation":
            # Method 3: 随机reward（不影响选择）
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
    """计算某张图片与所有其他图片的余弦相似度"""
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
    """可视化选中的图片"""
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
    运行单次实验
    
    Args:
        method: "eeg_guidance", "target_image_guidance", "random_generation"
        target_idx: 目标图像索引
        seed: 随机种子
        config: 配置字典
    
    Returns:
        results: 包含各种评估指标的字典
    """
    print(f"\n{'='*80}")
    print(f"Method: {method} | Target: {target_idx} | Seed: {seed}")
    print(f"{'='*80}")
    
    # 设置随机种子
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 验证索引
    assert target_idx < len(image_list), f"target_idx超出图片数量范围({len(image_list)})"
    assert target_idx < len(embed_list), f"target_idx超出embed数量范围({len(embed_list)})"
    
    target_image_path = os.path.join(image_dir, image_list[target_idx])
    target_eeg_embed_path = os.path.join(embed_dir, embed_list[target_idx])
    print(f"当前索引: {target_idx}")
    print(f"图片: {target_image_path}")
    print(f"eeg_embed: {target_eeg_embed_path}")
    
    # 加载目标特征
    target_eeg_feature = torch.load(target_eeg_embed_path, weights_only=False)
    
    # 加载目标图像的CLIP embedding
    target_image_pil = Image.open(target_image_path).convert("RGB")
    with torch.no_grad():
        target_clip_embed = vlmodel.encode_image(preprocess_train(target_image_pil).unsqueeze(0).to(device))
    
    # 根据方法选择优化目标
    if method == "eeg_guidance":
        target_feature = target_eeg_feature  # 使用EEG feature
    elif method == "target_image_guidance":
        target_feature = target_clip_embed  # 使用CLIP embedding
    elif method == "random_generation":
        target_feature = target_eeg_feature  # 随机方法不使用，但需要占位
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 其余参数
    sub = 'sub-01'
    fs = 250
    selected_channel_idxes = slice(None)
    dnn = 'alexnet'
    encoding_model_path = f'/home/ldy/Workspace/Closed_loop_optimizing/kyw/closed-loop/EEG-encoding/EEG_encoder/results/{sub}/synthetic_eeg_data/encoding-end_to_end/dnn-{dnn}/modeled_time_points-all/pretrained-True/lr-1e-05__wd-0e+00__bs-064/model_state_dict.pt'
    f_encoder = "/home/ldy/Workspace/Closed_loop_optimizing/kyw/closed-loop/sub_model/sub-01/diffusion_alexnet/pretrained_True/gene_gene/ATM_S_reconstruction_scale_0_1000_40.pth"
    checkpoint = torch.load(f_encoder, map_location=device, weights_only=False)
    eeg_model = ATMS()
    eeg_model.load_state_dict(checkpoint['eeg_model_state_dict'])
    
    # 获取图像池
    test_images_path = get_image_pool(image_dir)
    print(f"test_images_path {len(test_images_path)}")
    
    if target_image_path in test_images_path:
        test_images_path.remove(target_image_path)
    
    # 使用外部传入的test_set_img_embeds（避免重复加载）
    test_set_img_embeds = config['test_set_img_embeds']
    
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_save_dir = os.path.join(config['save_path'], timestamp, method, f'target_{target_idx}_seed_{seed}')
    os.makedirs(exp_save_dir, exist_ok=True)
    plots_save_folder = os.path.join(exp_save_dir, 'plots')
    os.makedirs(plots_save_folder, exist_ok=True)
    
    # 实验开始前强制清理显存
    torch.cuda.empty_cache()
    
    # 初始化Generator（第一次加载IP adapter，后续实验不重新加载）
    load_ip = config.get('first_run', False)
    if load_ip:
        config['first_run'] = False  # 标记已加载
    
    # 从config获取最小数据量阈值
    min_data_threshold = config.get('min_data_threshold', 10)
    Generator = HeuristicGenerator(pipe, vlmodel, preprocess_train, device=device, seed=seed, 
                                   load_ip_adapter=load_ip, min_data_threshold=min_data_threshold)
    
    # 迭代优化
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
    
    # 初始化最后一轮数据（用于最终评估）
    final_loop_images = []
    final_loop_eegs = []
    final_loop_rewards = []
    final_loop_losses = []
    final_loop_eeg_scores = []  # 用于基于EEG score选择Top-5
    
    for t in range(num_loops):
        print(f"Loop {t + 1}/{num_loops}")
        loop_save_dir = os.path.join(exp_save_dir, f'loop_{t+1}')
        os.makedirs(loop_save_dir, exist_ok=True)
        loop_sample_ten = []
        loop_reward_ten = []
        loop_eeg_ten = []
        loop_loss_ten = []
        
        if method == "random_generation":
            # Method 3: Random Generation (Null baseline) - 生成随机图像
            print("Method 3: Random Generation (Null baseline) - Generating random images...")
            
            # 只在第一轮执行，后续轮不需要
            if t == 0:
                # 创建临时Generator用于随机生成（不加载IP adapter以节省内存）
                print("Creating temporary generator for random image generation...")
                temp_generator = torch.Generator(device=device).manual_seed(seed)
                
                # 生成5张随机图像
                NUM_RANDOM_IMAGES = 5
                
                # 创建随机噪声
                epsilon = torch.randn(
                    Generator.num_inference_steps + 1, 
                    NUM_RANDOM_IMAGES,
                    Generator.pipe.unet.config.in_channels,
                    Generator.pipe.unet.config.sample_size,
                    Generator.pipe.unet.config.sample_size,
                    device=device, 
                    generator=temp_generator
                )
                
                # 随机初始化 pseudo target（不进行优化）
                random_pseudo_target = torch.randn(
                    NUM_RANDOM_IMAGES, 
                    Generator.dimension, 
                    device=device, 
                    generator=temp_generator
                )
                
                print(f"Generating {NUM_RANDOM_IMAGES} random images without optimization...")
                # 直接生成，不经过优化循环
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
                
                # 逐张解码以避免OOM
                chosen_images = []
                for i in range(NUM_RANDOM_IMAGES):
                    img = Generator.latents_to_images(latents[i:i+1])  # 每次只解码1张
                    chosen_images.extend(img)
                    del img
                    if i % 2 == 0:  # 每2张清理一次
                        torch.cuda.empty_cache()
                
                # 计算EEG（用于计算EEG score）
                synthetic_eegs = generate_eeg_from_image(encoding_model_path, chosen_images, device)
                chosen_eegs = synthetic_eegs
                
                # 计算reward（这里只是为了保持接口一致，实际上random baseline不需要reward）
                chosen_rewards = []
                for idx, eeg in enumerate(synthetic_eegs):
                    # 使用真实的EEG相似度作为记录（虽然不用于优化）
                    cs, _ = reward_function_clip_embed(eeg, eeg_model, target_eeg_feature, sub, device)
                    chosen_rewards.append(cs)
                
                chosen_losses = [0] * len(chosen_images)
                
                loop_sample_ten.extend(chosen_images)
                loop_eeg_ten.extend(chosen_eegs)
                loop_reward_ten.extend(chosen_rewards)
                loop_loss_ten.extend(chosen_losses)
                
                # 清理
                del epsilon, random_pseudo_target, latents, synthetic_eegs, temp_generator
                torch.cuda.empty_cache()
                
                print(f"Random baseline rewards: {[f'{r:.4f}' for r in chosen_rewards]}")
                print(f"Random baseline mean reward: {np.mean(chosen_rewards):.4f}")
                
                # 对于random baseline，保存数据用于最终评估
                final_loop_images = loop_sample_ten.copy()
                final_loop_eegs = loop_eeg_ten.copy()
                final_loop_rewards = loop_reward_ten.copy()
                final_loop_losses = loop_loss_ten.copy()
                final_loop_eeg_scores = chosen_rewards.copy()  # Random的rewards就是EEG scores
                
                # 对于random baseline，第一轮就结束，直接break
                # 不需要进行多轮迭代
                print("Random generation baseline completed (single round generation)")
            else:
                # 不应该进入这里，因为random_generation只需要第一轮
                pass
        
        else:
            # Method 1 & 2: 正常的优化流程
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
                
                # 清理临时变量
                del tensor_loop_sample_ten, tensor_loop_sample_ten_embeds
                torch.cuda.empty_cache()
            
            else:
                # 图像融合生成（每轮多次融合）
                tensor_fit_images = [preprocess_train(i) for i in fit_images]    
                with torch.no_grad():
                    img_embeds = vlmodel.encode_image(torch.stack(tensor_fit_images).to(device))    
                
                # 清理临时tensor
                del tensor_fit_images
                torch.cuda.empty_cache()
                
                # 🔥 关键修改：每轮进行多次融合
                NUM_FUSIONS_PER_ROUND = 2  # 从1次增加到2次
                print(f"[INFO] Performing {NUM_FUSIONS_PER_ROUND} fusions in this round")
                
                all_generated_images = []
                all_fusion_source_images = []
                
                for fusion_idx in range(NUM_FUSIONS_PER_ROUND):
                    print(f"  Fusion {fusion_idx + 1}/{NUM_FUSIONS_PER_ROUND}...")
                    
                    # 融合生成，同时返回被用于融合的原始图像索引
                    generated_images, (idx1, idx2) = fusion_image_to_images(Generator, img_embeds, fit_rewards, device, 512, save_path=loop_save_dir)
                    synthetic_eegs = generate_eeg_from_image(encoding_model_path, generated_images, device)
                    
                    # 添加融合生成的图像
                    loop_sample_ten.extend(generated_images)
                    loop_eeg_ten.extend(synthetic_eegs)
                    all_generated_images.extend(generated_images)
                    
                    # 计算融合生成图像的reward
                    for idx, eeg in enumerate(synthetic_eegs):
                        if method == "eeg_guidance":
                            cs, eeg_feature = reward_function_clip_embed(eeg, eeg_model, target_feature, sub, device)
                        elif method == "target_image_guidance":
                            cs = reward_function_clip_embed_image(generated_images[idx], target_feature, vlmodel, preprocess_train, device)
                        loss = 0
                        loop_reward_ten.append(cs)
                        loop_loss_ten.append(loss)
                    
                    # 清理EEG数据
                    del synthetic_eegs
                    torch.cuda.empty_cache()
                    
                    # 添加用于融合的原始图像
                    print(f"  Adding fusion source images (idx {idx1} and {idx2})")
                    fusion_source_images = [fit_images[idx1], fit_images[idx2]]
                    fusion_source_eegs = generate_eeg_from_image(encoding_model_path, fusion_source_images, device)
                    
                    loop_sample_ten.extend(fusion_source_images)
                    loop_eeg_ten.extend(fusion_source_eegs)
                    all_fusion_source_images.extend(fusion_source_images)
                    
                    # 计算融合源图像的reward
                    for idx, eeg in enumerate(fusion_source_eegs):
                        if method == "eeg_guidance":
                            cs, eeg_feature = reward_function_clip_embed(eeg, eeg_model, target_feature, sub, device)
                        elif method == "target_image_guidance":
                            cs = reward_function_clip_embed_image(fusion_source_images[idx], target_feature, vlmodel, preprocess_train, device)
                        loss = 0
                        loop_reward_ten.append(cs)
                        loop_loss_ten.append(loss)
                    
                    # 清理
                    del fusion_source_eegs
                    torch.cuda.empty_cache()
                
                print(f"[INFO] Fusion complete: generated {len(all_generated_images)} new images, added {len(all_fusion_source_images)} source images")
                
                # 贪婪采样（改进版）：基于target而不是生成图像进行采样
                # 🔥 关键修改：使用 target_clip_embed 而不是 img_embeds
                greedy_images = []
                TOP_K = 20  # 增加top-k范围，提高多样性
                # 贪婪采样数量与融合生成数量保持一致
                NUM_GREEDY_SAMPLES = len(all_generated_images)  # 每轮融合生成的总图像数
                
                # 获取可用图像的索引
                available_indices = []
                for i, path in enumerate(test_images_path):
                    if path not in processed_paths:
                        available_indices.append(i)
                
                if len(available_indices) > 0:
                    # 🔥 关键修改：基于 target_clip_embed 计算相似度
                    available_features = test_set_img_embeds[available_indices]
                    cosine_similarities = compute_embed_similarity(
                        target_clip_embed.to(device), 
                        available_features.to(device)
                    )
                    sorted_available_indices = np.argsort(cosine_similarities.cpu())
                    
                    # 从 top-K 中随机采样多张图像
                    top_indices = sorted_available_indices[-min(TOP_K, len(sorted_available_indices)):]
                    
                    # 从 top-K 中随机选择 NUM_GREEDY_SAMPLES 张（不重复）
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
                
                # 清理EEG和img_embeds
                del synthetic_eegs, img_embeds
                torch.cuda.empty_cache()
                
                # 选择top-4
                loop_probabilities = softmax(loop_reward_ten)
                chosen_rewards, chosen_losses, chosen_images, chosen_eegs = select_from_images(
                    loop_probabilities, loop_reward_ten, loop_loss_ten, loop_sample_ten, loop_eeg_ten, size=4)
                
                # 按reward排序
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
                
                # 清理临时变量（Method 1 & 2分支）
                del tensor_loop_sample_ten, tensor_loop_sample_ten_embeds
                torch.cuda.empty_cache()
        
        # 更新fit数据
        fit_images = chosen_images
        fit_eegs = chosen_eegs
        fit_rewards = chosen_rewards
        fit_losses = chosen_losses
        
        all_chosen_rewards.extend(chosen_rewards)
        all_chosen_losses.extend(chosen_losses)
        all_chosen_images.extend(chosen_images)
        all_chosen_eegs.extend(chosen_eegs)
        
        # 可视化
        visualize_top_images(loop_sample_ten, loop_reward_ten, loop_save_dir, t)
        
        # 记录历史最佳
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
        
        # Random baseline只需要一轮，直接结束
        if method == "random_generation" and t == 0:
            print("Random baseline completed after first round.")
            break
        
        # 早停检查
        if len(history_cs) >= 2:
            if history_cs[-1] != history_cs[-2]:
                diff = abs(history_cs[-1] - history_cs[-2])
                print(f"History: {history_cs[-1]:.4f}, {history_cs[-2]:.4f}, diff={diff:.6f}")
                if diff <= 1e-4:
                    print("The difference is within 1e-4, stopping early.")
                    break
        
        # 保存最后一轮的数据用于最终评估
        if t == num_loops - 1:  # 最后一轮
            final_loop_images = loop_sample_ten.copy()
            final_loop_eegs = loop_eeg_ten.copy()
            final_loop_rewards = loop_reward_ten.copy()
            final_loop_losses = loop_loss_ten.copy()
            
            # 计算真实的EEG scores用于最终选择（无论优化目标是什么）
            print(f"\n[INFO] Computing EEG scores for final selection...")
            final_loop_eeg_scores = []
            for eeg in loop_eeg_ten:
                eeg_score, _ = reward_function_clip_embed(eeg, eeg_model, target_eeg_feature, sub, device)
                final_loop_eeg_scores.append(eeg_score)
            print(f"Last round EEG scores: {[f'{s:.4f}' for s in final_loop_eeg_scores]}")
        
        # 清理每轮循环的临时数据
        del loop_sample_ten, loop_eeg_ten, loop_reward_ten, loop_loss_ten
        torch.cuda.empty_cache()
    
    # 绘制相似度曲线
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
    
    # === 评估阶段 ===
    # 最终评估：基于EEG score选择Top-5（确定性）
    # 原理：无论优化策略如何，最终目标都是产生期望的EEG feature
    if method == "random_generation":
        # Random baseline使用5张图像
        final_images = chosen_images  # 应该是5张
        final_chosen_eegs = chosen_eegs  # 对应的EEG
        print(f"\nRandom baseline: evaluating all {len(final_images)} images (no selection)")
    else:
        # 其他方法：从最后一轮的所有候选（8张）中基于EEG score选择 Top-5
        if len(final_loop_images) > 0:
            print(f"\n[INFO] Final evaluation: Selecting Top-5 based on EEG scores from last round's {len(final_loop_images)} candidates")
            print(f"Last round optimization rewards ({method}): {[f'{r:.4f}' for r in final_loop_rewards]}")
            
            # 关键：基于EEG score（而非优化reward）选择Top-5
            # 这样确保评估的是真正对EEG feature最优的图像
            final_chosen_eeg_scores, final_chosen_losses, final_images, final_chosen_eegs = select_top_k_images(
                final_loop_eeg_scores, final_loop_losses, final_loop_images, final_loop_eegs, size=5)
            
            print(f"Selected Top-5 EEG scores: {[f'{s:.4f}' for s in final_chosen_eeg_scores]}")
            print(f"{method}: evaluating {len(final_images)} images (Top-5 by EEG score)")
        else:
            # 如果提前停止，使用最后一次选中的图像
            print(f"[WARNING] No final loop data available, using last chosen images")
            final_images = chosen_images
            final_chosen_eegs = chosen_eegs
            print(f"{method}: evaluating {len(final_images)} images")
    
    # 保存最终图像
    saved_image_paths = []
    for i, img in enumerate(final_images):
        save_name = os.path.join(exp_save_dir, f"final_generated_{i}.png")
        img.save(save_name)
        saved_image_paths.append(save_name)
    
    # 计算评估指标
    # 统一策略：所有方法都报告 EEG score (primary) 和 CLIP score (secondary)
    # 原理：任务目标是优化视觉刺激产生期望的EEG feature
    
    print("\n" + "="*80)
    print("FINAL EVALUATION METRICS")
    print("="*80)
    
    # 1. 计算 EEG scores (Primary Metric - 任务的真正目标)
    print("\n[Primary Metric] Computing EEG similarity...")
    if method != "random_generation" and len(final_loop_images) > 0:
        # 优化方法：已经有了final_chosen_eegs，直接使用
        eeg_scores = []
        for eeg in final_chosen_eegs:
            score, _ = reward_function_clip_embed(eeg, eeg_model, target_eeg_feature, sub, device)
            eeg_scores.append(score)
    else:
        # Random baseline或提前停止：需要重新计算
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
    
    # 2. 计算 CLIP scores (Secondary Metric - 参考指标)
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
    
    # 彻底清理所有大对象
    # 先清理Generator内部的模型
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
    
    # 强制清理显存和内存
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # 返回结果
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
    """生成benchmark报告
    
    注意：
    - random_generation使用5张随机采样的图像（真正的无偏baseline）
    - eeg_guidance和target_image_guidance使用4张优化后的图像
    """
    save_path = config['save_path']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 创建DataFrame
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
                'Num_Images': len(result['eeg_scores'])  # 记录图像数量
            })
    
    df = pd.DataFrame(records)
    
    # 保存详细结果
    csv_path = os.path.join(save_path, f'detailed_results_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n详细结果已保存到: {csv_path}")
    
    # 2. 计算汇总统计
    summary_stats = df.groupby('Method').agg({
        'EEG_Score': ['mean', 'std', 'min', 'max'],
        'CLIP_Score': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    summary_path = os.path.join(save_path, f'summary_statistics_{timestamp}.csv')
    summary_stats.to_csv(summary_path)
    print(f"汇总统计已保存到: {summary_path}")
    
    # 打印汇总统计
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY STATISTICS")
    print(f"{'='*80}")
    print("\n注意：")
    print("  - random_generation: 使用5张随机生成的图像（无优化的null baseline）")
    print("  - eeg_guidance: 使用4张基于EEG特征优化的图像，主要评估指标是EEG Score")
    print("  - target_image_guidance: 使用4张基于目标图像CLIP特征优化的图像，主要评估指标是CLIP Score")
    print("\n评估说明：")
    print("  - EEG Score: 生成图像的EEG特征 vs 目标图像的EEG特征（eeg_guidance的优化目标）")
    print("  - CLIP Score: 生成图像的CLIP embedding vs 目标图像的CLIP embedding（target_image_guidance的优化目标）")
    print("  - random_generation: 使用随机噪声生成图像，不做任何优化（与Offline方法的baseline一致）")
    print(f"\n{'='*80}")
    print(summary_stats)
    print(f"{'='*80}\n")
    
    # 统计每个方法的图像数量
    print("每个方法的图像数量统计:")
    for method in df['Method'].unique():
        method_df = df[df['Method'] == method]
        num_images_per_exp = method_df.groupby(['Target_Idx', 'Seed']).size().unique()
        print(f"  {method}: {num_images_per_exp[0] if len(num_images_per_exp) == 1 else num_images_per_exp} 张图像/实验")
    print()
    
    # 3. 生成可视化对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # EEG Score对比
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
    
    # CLIP Score对比
    sns.boxplot(data=df, x='Method_Label', y='CLIP_Score', ax=axes[1])
    axes[1].set_title('CLIP Similarity Score', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Method', fontsize=12)
    axes[1].set_ylabel('CLIP Score', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, f'benchmark_comparison_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存到: {plot_path}")
    plt.close()
    
    # 4. 生成条形图对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    summary_mean = df.groupby('Method')[['EEG_Score', 'CLIP_Score']].mean()
    summary_std = df.groupby('Method')[['EEG_Score', 'CLIP_Score']].std()
    
    methods = list(method_names.keys())
    method_labels = [method_names[m] for m in methods]
    
    # EEG Score条形图
    eeg_means = [summary_mean.loc[m, 'EEG_Score'] if m in summary_mean.index else 0 for m in methods]
    eeg_stds = [summary_std.loc[m, 'EEG_Score'] if m in summary_std.index else 0 for m in methods]
    axes[0].bar(method_labels, eeg_means, yerr=eeg_stds, capsize=5, alpha=0.7, color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[0].set_title('EEG Similarity (Mean ± Std)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('EEG Score', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # CLIP Score条形图
    clip_means = [summary_mean.loc[m, 'CLIP_Score'] if m in summary_mean.index else 0 for m in methods]
    clip_stds = [summary_std.loc[m, 'CLIP_Score'] if m in summary_std.index else 0 for m in methods]
    axes[1].bar(method_labels, clip_means, yerr=clip_stds, capsize=5, alpha=0.7, color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[1].set_title('CLIP Similarity (Mean ± Std)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('CLIP Score', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    bar_plot_path = os.path.join(save_path, f'benchmark_barplot_{timestamp}.png')
    plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
    print(f"条形图已保存到: {bar_plot_path}")
    plt.close()
    
    # 5. 保存JSON格式的完整结果
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
    
    # 创建可序列化的config副本（排除Tensor对象）
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
    print(f"完整结果JSON已保存到: {json_path}")
    
    # 6. 生成专门的Chance Level分析报告
    if 'random_generation' in df['Method'].unique():
        random_df = df[df['Method'] == 'random_generation']
        
        chance_level_report = []
        chance_level_report.append("=" * 80)
        chance_level_report.append("CHANCE LEVEL (Random Baseline) 详细分析")
        chance_level_report.append("=" * 80)
        chance_level_report.append("")
        chance_level_report.append("方法说明：")
        chance_level_report.append("  使用随机噪声生成5张图像（不考虑任何目标特征）")
        chance_level_report.append("  不包含任何优化或引导过程")
        chance_level_report.append("  使用随机初始化的 pseudo target embedding")
        chance_level_report.append("  与 Offline Generation 的 random baseline 实现一致")
        chance_level_report.append("")
        chance_level_report.append("统计结果：")
        chance_level_report.append(f"  EEG Score:")
        chance_level_report.append(f"    均值: {random_df['EEG_Score'].mean():.4f}")
        chance_level_report.append(f"    标准差: {random_df['EEG_Score'].std():.4f}")
        chance_level_report.append(f"    最小值: {random_df['EEG_Score'].min():.4f}")
        chance_level_report.append(f"    最大值: {random_df['EEG_Score'].max():.4f}")
        chance_level_report.append(f"    中位数: {random_df['EEG_Score'].median():.4f}")
        chance_level_report.append("")
        chance_level_report.append(f"  CLIP Score:")
        chance_level_report.append(f"    均值: {random_df['CLIP_Score'].mean():.4f}")
        chance_level_report.append(f"    标准差: {random_df['CLIP_Score'].std():.4f}")
        chance_level_report.append(f"    最小值: {random_df['CLIP_Score'].min():.4f}")
        chance_level_report.append(f"    最大值: {random_df['CLIP_Score'].max():.4f}")
        chance_level_report.append(f"    中位数: {random_df['CLIP_Score'].median():.4f}")
        chance_level_report.append("")
        chance_level_report.append(f"  样本数: {len(random_df)} 张图像")
        chance_level_report.append(f"  实验数: {random_df.groupby(['Target_Idx', 'Seed']).ngroups} 次")
        chance_level_report.append("")
        chance_level_report.append("与其他方法的对比：")
        chance_level_report.append("  注意：不同方法的优化目标不同")
        chance_level_report.append("  - eeg_guidance: 优化目标是EEG Score（主要指标）")
        chance_level_report.append("  - target_image_guidance: 优化目标是CLIP Score（主要指标）")
        chance_level_report.append("")
        
        for method in ['eeg_guidance', 'target_image_guidance']:
            if method in df['Method'].unique():
                method_df = df[df['Method'] == method]
                eeg_improvement = (method_df['EEG_Score'].mean() - random_df['EEG_Score'].mean()) / random_df['EEG_Score'].mean() * 100
                clip_improvement = (method_df['CLIP_Score'].mean() - random_df['CLIP_Score'].mean()) / random_df['CLIP_Score'].mean() * 100
                
                if method == 'eeg_guidance':
                    chance_level_report.append(f"  {method}:")
                    chance_level_report.append(f"    EEG Score提升: {eeg_improvement:+.2f}% ← 主要优化目标")
                    chance_level_report.append(f"    CLIP Score提升: {clip_improvement:+.2f}%")
                else:  # target_image_guidance
                    chance_level_report.append(f"  {method}:")
                    chance_level_report.append(f"    EEG Score提升: {eeg_improvement:+.2f}%")
                    chance_level_report.append(f"    CLIP Score提升: {clip_improvement:+.2f}% ← 主要优化目标")
                chance_level_report.append("")
        
        chance_level_report.append("=" * 80)
        
        chance_level_text = "\n".join(chance_level_report)
        print("\n" + chance_level_text)
        
        # 保存到文件
        chance_level_path = os.path.join(save_path, f'chance_level_analysis_{timestamp}.txt')
        with open(chance_level_path, 'w') as f:
            f.write(chance_level_text)
        print(f"\nChance Level分析已保存到: {chance_level_path}")
    
    return df, summary_stats


def main():
    """主函数：运行三种方法的benchmark"""
    
    # 初始化模型（共享）
    print("Loading shared models...")
    model_type = 'ViT-H-14'
    vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
        model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device=device)
    vlmodel.to(device)
    
    generator = Generator4Embeds(device=device)
    pipe = generator.pipe
    
    # 加载test set image embeds（只加载一次，所有实验共享）
    print("Loading test set image embeddings...")
    test_set_img_embeds = torch.load("/mnt/dataset1/ldy/Workspace/FLORA/data_preparing/ViT-H-14_features_test.pt")['img_features'].cpu()
    print(f"Loaded {test_set_img_embeds.shape[0]} image embeddings")
    
    # 配置参数
    config = {
        # 'target_indices': [i for i in range(30)],  # 要测试的目标图像索引
        'target_indices': np.linspace(1, 200, 10, dtype=int).tolist(),
        'num_seeds': 1,  # 每个方法运行的随机种子数
        'num_loops': 10,  # 每次实验的迭代轮数
        'save_path': '/home/ldy/Workspace/Closed_loop_optimizing/outputs/benchmark_heuristic_generation',
        'methods': ['eeg_guidance', 'target_image_guidance', 'random_generation'],
        'test_set_img_embeds': test_set_img_embeds,  # 共享的image embeddings
        'first_run': True,  # 标记第一次运行，需要加载IP adapter
        'min_data_threshold': 10  # 最小数据量阈值，低于此值将跳过优化
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
    
    # 运行所有实验
    for target_idx in config['target_indices']:
        for method in config['methods']:
            for seed in range(config['num_seeds']):
                try:
                    # 实验前强制清理
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    result = run_single_experiment(method, target_idx, seed, config, vlmodel, preprocess_train, pipe)
                    all_results.append(result)
                    
                    # 实验后强制清理
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"\n!!! Error in {method}, target {target_idx}, seed {seed}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # 错误后强制清理
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
    
    # 清理共享的test_set_img_embeds
    if 'test_set_img_embeds' in config:
        del config['test_set_img_embeds']
    torch.cuda.empty_cache()
    
    # 生成报告
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
    
    # 最终清理
    del vlmodel, pipe, generator
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

