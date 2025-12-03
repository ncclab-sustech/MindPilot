"""
Benchmark for Offline Generation with Three Different Methods:
1. EEG Feature Guidance (MindPilot) - 使用目标脑电特征作为优化目标
2. Target Image Guidance (Ceiling) - 使用目标图像的CLIP embedding作为优化目标
3. Random Generation (Null) - 完全随机生成，不做任何优化（null baseline）

评估方式：
所有方法都使用相同的评估标准：
- EEG Score: 生成图像 vs 目标图像的脑电特征相似度
- CLIP Score: 生成图像 vs 目标图像的CLIP embedding相似度 
./run_benchmark.sh
"""

import os
import sys

import json
import pandas as pd
from datetime import datetime

# proxy = 'http://127.0.0.1:7881'
# os.environ['http_proxy'] = proxy
# os.environ['https_proxy'] = proxy

import numpy as np
import torch
import random
from PIL import Image
import open_clip
import torch.nn.functional as F
import torchvision.transforms as transforms

import seaborn as sns

sys.path.append('/home/ldy/Workspace/Closed_loop_optimizing')
sys.path.append('/home/ldy/Workspace/Closed_loop_optimizing/model')

from model.utils import load_model_encoder, generate_eeg
from model.custom_pipeline_low_level import Generator4Embeds
from model.ATMS_retrieval import ATMS, get_eeg_features
from model.pseudo_target_model import PseudoTargetModel
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 模型初始化 ---
print("Loading models...")
model_type = 'ViT-H-14'
vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
    model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device=device)
vlmodel.to(device)

generator = Generator4Embeds(device=device)
pipe = generator.pipe

# --- 目录配置 ---
image_dir = '/home/ldy/Workspace/Closed_loop_optimizing/test_images'
embed_dir = '/home/ldy/Workspace/Closed_loop_optimizing/data/clip_embed/2025-09-21'

image_list = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
embed_list = sorted([f for f in os.listdir(embed_dir) if f.endswith('_embed.pt')])


class HeuristicGenerator:
    def __init__(self, pipe, vlmodel, preprocess_train, device="cuda", seed=0):
        self.pipe = pipe
        self.vlmodel = vlmodel
        self.preprocess_train = preprocess_train
        self.device = device
        self.seed = seed

        # Hyperparameters
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

        # Initialize components
        self.pseudo_target_model = PseudoTargetModel(dimension=self.dimension, noise_level=1e-4).to(self.device)
        self.generator = torch.Generator(device=device).manual_seed(self.seed)

        # Load IP adapter
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter", subfolder="sdxl_models", 
            weight_name="ip-adapter_sdxl_vit-h.bin", 
            torch_dtype=torch.bfloat16)
        self.pipe.set_ip_adapter_scale(0.5)

    def latents_to_images(self, latents):
        shift_factor = self.pipe.vae.config.shift_factor if self.pipe.vae.config.shift_factor else 0.0
        latents = (latents / self.pipe.vae.config.scaling_factor) + shift_factor
        images = self.pipe.vae.decode(latents, return_dict=False)[0]
        images = self.pipe.image_processor.postprocess(images.detach())
        return images
    
    def generate(self, data_x, data_y, tar_image_embed, prompt='', save_path=None, start_embedding=None):
        # Initialize noise
        epsilon = torch.randn(self.num_inference_steps+1, self.generate_batch_size, 
                            self.pipe.unet.config.in_channels, 
                            self.pipe.unet.config.sample_size, 
                            self.pipe.unet.config.sample_size, 
                            device=self.device, generator=self.generator)

        # Initialize pseudo target
        if start_embedding is not None:
            pseudo_target = start_embedding.expand(self.generate_batch_size, self.dimension).to(self.device)
        else:
            pseudo_target = torch.randn(self.generate_batch_size, self.dimension, device=self.device, generator=self.generator)

        # Optimization loop
        for step in range(self.total_steps):
            current_data_x, current_data_y = self.pseudo_target_model.get_model_data()   
            if current_data_y.size(0) < 5: 
                print("Data insufficient, returning random generation.")
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

            step_size = self.initial_step_size / (1 + self.decay_rate * step)
            pseudo_target, _ = self.pseudo_target_model.estimate_pseudo_target(pseudo_target, step_size=step_size) 

        # Final Generation
        final_images = self.latents_to_images(self.pipe(
                [prompt]*self.generate_batch_size,
                ip_adapter_image_embeds=[pseudo_target.unsqueeze(0).type(torch.bfloat16).to(self.device)],
                latents=epsilon[0].type(torch.bfloat16),
                given_noise=epsilon[1:].type(torch.bfloat16),
                output_type="latent",
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                eta=1.0,
            ).images)
        
        return final_images


def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)


def generate_eeg_from_image_paths(model_path, test_image_list, device):
    synthetic_eegs = []
    model = load_model_encoder(model_path, device)
    for idx, image_path in enumerate(test_image_list):
        image_tensor = preprocess_image(image_path, device)
        synthetic_eeg = generate_eeg(model, image_tensor, device)
        synthetic_eegs.append(synthetic_eeg)
        del image_tensor
    del model
    torch.cuda.empty_cache()
    return np.asarray(synthetic_eegs)


def generate_eeg_from_images(model_path, images, device):
    """从PIL图像列表生成EEG"""
    def preprocess_gen_img(img, dev):
        t = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return t(img).unsqueeze(0).to(dev)

    synthetic_eegs = []
    model = load_model_encoder(model_path, device)
    for img in images:
        img_tensor = preprocess_gen_img(img, device)
        synthetic_eeg = generate_eeg(model, img_tensor, device)
        synthetic_eegs.append(synthetic_eeg)
        del img_tensor
    
    del model
    torch.cuda.empty_cache()
    return synthetic_eegs


def compute_eeg_similarity(eeg, eeg_model, target_feature, sub, device):
    """计算EEG相似度"""
    eeg_feature = get_eeg_features(eeg_model, torch.tensor(eeg).unsqueeze(0), device, sub)    
    similarity = torch.nn.functional.cosine_similarity(eeg_feature.to(device), target_feature.to(device))
    similarity = (similarity + 1) / 2
    return similarity.item()


def compute_clip_similarity(images, target_clip_embed, vlmodel, preprocess_train, device):
    """计算CLIP相似度"""
    tensor_images = [preprocess_train(img) for img in images]
    with torch.no_grad():
        image_embeds = vlmodel.encode_image(torch.stack(tensor_images).to(device))
        # 归一化
        image_embeds = F.normalize(image_embeds, dim=-1)
        target_clip_embed = F.normalize(target_clip_embed, dim=-1)
        similarities = F.cosine_similarity(image_embeds, target_clip_embed)
        similarities = (similarities + 1) / 2
    return similarities.cpu().numpy()


def run_single_experiment(method, target_idx, seed, config):
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
    
    # 加载目标数据
    target_image_path = os.path.join(image_dir, image_list[target_idx])
    target_eeg_embed_path = os.path.join(embed_dir, embed_list[target_idx])
    target_eeg_feature = torch.load(target_eeg_embed_path, weights_only=False)
    
    # 加载目标图像的CLIP embedding
    target_image_pil = Image.open(target_image_path).convert("RGB")
    with torch.no_grad():
        target_clip_embed = vlmodel.encode_image(preprocess_train(target_image_pil).unsqueeze(0).to(device))
    
    # 加载EEG模型
    sub = 'sub-01'
    dnn = 'alexnet'
    f_encoder = "/home/ldy/Workspace/Closed_loop_optimizing/kyw/closed-loop/sub_model/sub-01/diffusion_alexnet/pretrained_True/gene_gene/ATM_S_reconstruction_scale_0_1000_40.pth"
    checkpoint = torch.load(f_encoder, map_location=device, weights_only=False)
    eeg_model = ATMS()
    eeg_model.load_state_dict(checkpoint['eeg_model_state_dict'])
    encoding_model_path = f'/home/ldy/Workspace/Closed_loop_optimizing/kyw/closed-loop/EEG-encoding/EEG_encoder/results/{sub}/synthetic_eeg_data/encoding-end_to_end/dnn-{dnn}/modeled_time_points-all/pretrained-True/lr-1e-05__wd-0e+00__bs-064/model_state_dict.pt'
    
    # 获取图像池
    def get_image_pool(image_set_path):
        test_images_path = []
        for root, dirs, files in os.walk(image_set_path):
            for file in sorted(files):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    test_images_path.append(os.path.join(root, file))
        return test_images_path
    
    test_images_path = get_image_pool(image_dir)
    if target_image_path in test_images_path:
        test_images_path.remove(target_image_path)
    
    print(f"Available images in pool (excluding target): {len(test_images_path)}")
    
    # 对于 random_generation 方法，直接跳转到生成部分
    if method == "random_generation":
        print("Method 3: Random Generation (Null) - Skipping offline data, pure random generation...")
        # 直接跳到 random_generation 的生成代码部分
        # 不需要采样、不需要计算 embeddings、不需要计算 rewards
        pass  # 将在后面的 if-elif 中处理
    else:
        # 采样离线数据（仅对需要优化的方法）
        OFFLINE_BATCH_SIZE = config['offline_batch_size']
        
        # 检查边界情况
        if len(test_images_path) == 0:
            raise ValueError(f"No images available for sampling! Image directory: {image_dir}")
        
        if len(test_images_path) < OFFLINE_BATCH_SIZE:
            print(f"⚠️  Warning: Requested {OFFLINE_BATCH_SIZE} images, but only {len(test_images_path)} available.")
            print(f"    Using all {len(test_images_path)} available images.")
            sampled_paths = test_images_path
            actual_batch_size = len(test_images_path)
        else:
            sampled_paths = random.sample(test_images_path, OFFLINE_BATCH_SIZE)
            actual_batch_size = OFFLINE_BATCH_SIZE
        
        print(f"Sampling {actual_batch_size} images for offline optimization...")
        
        sampled_images = [Image.open(p).convert("RGB") for p in sampled_paths]
        
        # 计算CLIP Embeddings
        print("Encoding images to CLIP space...")
        tensor_images = [preprocess_train(img) for img in sampled_images]
        with torch.no_grad():
            offline_embeds = vlmodel.encode_image(torch.stack(tensor_images).to(device))
    
    # 根据方法选择不同的reward计算方式
    if method == "eeg_guidance":
        # Method 1: EEG Feature Guidance (MindPilot)
        print("Method 1: EEG Feature Guidance - Computing EEG rewards...")
        synthetic_eegs = generate_eeg_from_image_paths(encoding_model_path, sampled_paths, device)
        offline_rewards = []
        for eeg in synthetic_eegs:
            r = compute_eeg_similarity(eeg, eeg_model, target_eeg_feature, sub, device)
            offline_rewards.append(r)
        del synthetic_eegs
        
    elif method == "target_image_guidance":
        # Method 2: Target Image Guidance (Ceiling)
        print("Method 2: Target Image Guidance - Computing CLIP rewards with target image...")
        offline_rewards = []
        for embed in offline_embeds:
            # 计算与目标图像CLIP embedding的相似度
            sim = F.cosine_similarity(embed.unsqueeze(0), target_clip_embed, dim=-1)
            sim = (sim + 1) / 2
            offline_rewards.append(sim.item())
            
    elif method == "random_generation":
        # Method 3: Random Generation (Null baseline)
        # 完全随机生成，不做任何优化
        print("Method 3: Random Generation (Null) - No optimization, pure random generation...")
        
        # 创建Generator（不进行优化训练）
        print(f"Initializing Generator with seed {seed}...")
        Generator = HeuristicGenerator(pipe, vlmodel, preprocess_train, device=device, seed=seed)
        
        # random_generation 分支没有创建 offline_embeds，不需要清理
        torch.cuda.empty_cache()
        
        # 直接随机生成（不加载任何优化数据）
        print("Generating random images without optimization...")
        # 创建随机噪声
        epsilon = torch.randn(
            Generator.num_inference_steps + 1, 
            Generator.generate_batch_size,
            Generator.pipe.unet.config.in_channels,
            Generator.pipe.unet.config.sample_size,
            Generator.pipe.unet.config.sample_size,
            device=device, 
            generator=Generator.generator
        )
        
        # 随机初始化 pseudo target（不进行优化）
        random_pseudo_target = torch.randn(
            Generator.generate_batch_size, 
            Generator.dimension, 
            device=device, 
            generator=Generator.generator
        )
        
        # 直接生成，不经过优化循环
        latents = Generator.pipe(
            [''] * Generator.generate_batch_size,
            ip_adapter_image_embeds=[random_pseudo_target.unsqueeze(0).type(torch.bfloat16).to(device)],
            latents=epsilon[0].type(torch.bfloat16),
            given_noise=epsilon[1:].type(torch.bfloat16),
            output_type="latent",
            num_inference_steps=Generator.num_inference_steps,
            guidance_scale=Generator.guidance_scale,
            eta=1.0,
        ).images
        
        final_images = Generator.latents_to_images(latents)
        del epsilon, random_pseudo_target, latents
        torch.cuda.empty_cache()
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 对于非随机生成的方法，执行优化
    if method != "random_generation":
        offline_rewards_tensor = torch.tensor(offline_rewards).to(device)
        print(f"Max Reward in offline batch: {max(offline_rewards):.4f}")
        torch.cuda.empty_cache()
        
        # 创建Generator并训练
        print(f"Initializing Generator with seed {seed}...")
        Generator = HeuristicGenerator(pipe, vlmodel, preprocess_train, device=device, seed=seed)
        Generator.pseudo_target_model = PseudoTargetModel(dimension=1024, noise_level=1e-4).to(device)
        Generator.pseudo_target_model.add_model_data(
            offline_embeds, 
            (-offline_rewards_tensor * Generator.reward_scaling_factor).to(device)
        )
        
        # 生成图像
        print("Running offline generation...")
        final_images = Generator.generate(
            data_x=offline_embeds,
            data_y=offline_rewards_tensor,
            tar_image_embed=None,
            prompt='', 
            save_path=None 
        )
        torch.cuda.empty_cache()
        
        # 清理优化相关数据
        del offline_embeds
        del offline_rewards_tensor
        torch.cuda.empty_cache()
    
    # 保存生成的图像
    save_dir = os.path.join(config['save_path'], method, f'target_{target_idx}')
    os.makedirs(save_dir, exist_ok=True)
    
    saved_image_paths = []
    final_images_copy = []
    for i, img in enumerate(final_images):
        save_name = os.path.join(save_dir, f"generated_seed{seed}_{i}.png")
        img.save(save_name)
        saved_image_paths.append(save_name)
        final_images_copy.append(Image.open(save_name).convert("RGB"))
    
    # 清理Generator
    print("清理 Generator 和离线数据...")
    del Generator
    del final_images
    # 对于非随机生成的方法，offline 数据已在前面清理
    if method == "random_generation":
        # random_generation 方法的 offline 数据已经提前清理
        pass
    torch.cuda.empty_cache()
    
    # 评估：计算EEG相似度
    print("Computing EEG similarity...")
    recon_eegs = generate_eeg_from_images(encoding_model_path, final_images_copy, device)
    eeg_scores = []
    for recon_eeg in recon_eegs:
        score = compute_eeg_similarity(recon_eeg, eeg_model, target_eeg_feature, sub, device)
        eeg_scores.append(score)
    del recon_eegs
    torch.cuda.empty_cache()
    
    # 评估：计算CLIP相似度
    print("Computing CLIP similarity...")
    clip_scores = compute_clip_similarity(final_images_copy, target_clip_embed, vlmodel, preprocess_train, device)
    
    # 清理
    del eeg_model
    del target_eeg_feature
    del target_clip_embed
    del final_images_copy
    torch.cuda.empty_cache()
    
    # 返回结果
    results = {
        'method': method,
        'target_idx': int(target_idx),  # 确保是 Python int
        'seed': int(seed),
        'eeg_scores': eeg_scores,  # 已经是 Python list of float
        'clip_scores': clip_scores.tolist(),  # 转换为 Python list
        'avg_eeg_score': float(np.mean(eeg_scores)),  # 转换为 Python float
        'avg_clip_score': float(np.mean(clip_scores)),  # 转换为 Python float
        'saved_paths': saved_image_paths
    }
    
    print(f"Results: EEG={results['avg_eeg_score']:.4f}, CLIP={results['avg_clip_score']:.4f}")
    return results


def generate_benchmark_report(all_results, config):
    """生成benchmark报告"""
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
                'CLIP_Score': clip_score
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
    print(summary_stats)
    print(f"{'='*80}\n")
    
    # 3. 生成可视化对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # EEG Score对比
    method_names = {
        'eeg_guidance': 'EEG Guidance\n(MindPilot)',
        'target_image_guidance': 'Target Image\n(Ceiling)',
        'random_generation': 'Random Generation\n(Null)'
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
    eeg_means = [summary_mean.loc[m, 'EEG_Score'] for m in methods]
    eeg_stds = [summary_std.loc[m, 'EEG_Score'] for m in methods]
    axes[0].bar(method_labels, eeg_means, yerr=eeg_stds, capsize=5, alpha=0.7, color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[0].set_title('EEG Similarity (Mean ± Std)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('EEG Score', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # CLIP Score条形图
    clip_means = [summary_mean.loc[m, 'CLIP_Score'] for m in methods]
    clip_stds = [summary_std.loc[m, 'CLIP_Score'] for m in methods]
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
    # 将 summary_stats 转换为可序列化的格式（展平多级列索引）
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
    
    json_results = {
        'config': config,
        'timestamp': timestamp,
        'summary_statistics': summary_dict,
        'all_results': all_results
    }
    
    json_path = os.path.join(save_path, f'complete_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"完整结果JSON已保存到: {json_path}")
    
    return df, summary_stats


def main():
    """主函数：运行三种方法的benchmark"""
    
    # 配置参数
    config = {
        # 'target_indices': [i for i in range(30)],  # 要测试的目标图像索引
        'target_indices': np.linspace(1, 200, 30, dtype=int).tolist(),
        'num_seeds': 1,  # 每个方法运行的随机种子数
        'offline_batch_size': 10,  # 离线数据批量大小
        'save_path': '/home/ldy/Workspace/Closed_loop_optimizing/outputs/benchmark_offline_generation',
        'methods': ['eeg_guidance', 'target_image_guidance', 'random_generation']
    }
    
    os.makedirs(config['save_path'], exist_ok=True)
    
    print(f"\n{'='*80}")
    print("OFFLINE GENERATION BENCHMARK")
    print(f"{'='*80}")
    print(f"Methods: {config['methods']}")
    print(f"Target indices: {config['target_indices']}")
    print(f"Seeds per method: {config['num_seeds']}")
    print(f"Offline batch size: {config['offline_batch_size']}")
    print(f"{'='*80}\n")
    
    all_results = []
    
    # 运行所有实验
    for target_idx in config['target_indices']:
        for method in config['methods']:
            for seed in range(config['num_seeds']):
                try:
                    result = run_single_experiment(method, target_idx, seed, config)
                    all_results.append(result)
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"\n!!! Error in {method}, target {target_idx}, seed {seed}: {e}")
                    import traceback
                    traceback.print_exc()
                    torch.cuda.empty_cache()
                    continue
    
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


if __name__ == '__main__':
    main()

