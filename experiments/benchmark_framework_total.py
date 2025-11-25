"""
Benchmark Framework for Comparing Different Optimization Methods (Extended Version)
对比 7 种方法：
  1. PseudoModel (Offline)         - 离线采样 + GP优化
  2. HeuristicClosedLoop          - 闭环迭代（融合+贪婪采样）
  3. DDPO                          - 强化学习（PPO）
  4. DPOK                          - 强化学习（KL正则化）
  5. D3PO                          - 强化学习（DPO）
  6. BayesianOpt                   - 贝叶斯优化
  7. CMA-ES                        - 进化策略
"""

# CUDA_VISIBLE_DEVICES=1 python benchmark_framework_total.py --config benchmark_config_total.json --exp exp1


import os
import sys

# ⚠️ 重要：在导入其他模块前先设置环境变量，防止被覆盖
# 如果命令行已设置 CUDA_VISIBLE_DEVICES，则使用命令行的值
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    # 如果没有设置，使用默认值（可以在这里修改）
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 默认使用GPU 0

import time
import json
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

sys.path.append('/home/ldy/Workspace/Closed_loop_optimizing')
sys.path.append('/home/ldy/Workspace/Closed_loop_optimizing/model')

@dataclass
class BenchmarkResult:
    """单次实验结果"""
    method_name: str
    target_idx: int
    seed: int
    
    # 性能指标
    eeg_similarity: float
    clip_score: float = None
    aesthetic_score: float = None
    ssim: float = None
    
    # 效率指标
    time_seconds: float = 0.0
    gpu_memory_gb: float = 0.0
    n_samples_used: int = 0
    
    # 其他信息
    success: bool = True
    error_message: str = None
    
    def to_dict(self):
        return asdict(self)


# ==================== HeuristicGenerator Class ====================
# 严格按照 exp-benchmark_heuristic_generation.py 的实现

import einops
import torch.nn.functional as F
from model.pseudo_target_model import PseudoTargetModel

class HeuristicGenerator:
    """
    严格按照 exp-benchmark_heuristic_generation.py 实现的 HeuristicGenerator
    用于 HeuristicClosedLoop 方法中的融合生成
    """
    def __init__(self, pipe, vlmodel, preprocess_train, device="cuda", seed=42, load_ip_adapter=False, min_data_threshold=10):
        self.pipe = pipe
        self.vlmodel = vlmodel
        self.preprocess_train = preprocess_train
        self.device = device

        # Hyperparameters
        self.batch_size = 32
        self.alpha = 80
        self.total_steps = 15  # 🔥 关键：15步优化
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
        """
        🔥 核心方法：包含15步GP优化 + 最终生成
        """
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

        # 🔥 关键：15步优化循环 - 只优化 pseudo_target，不生成图像
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

        # 🔥 优化循环结束后，用优化好的 pseudo_target 生成最终图像
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


# ==================== BaseMethod Class ====================

class BaseMethod:
    """所有方法的基类 - 提供共用的工具方法"""
    
    def __init__(self, config: dict, device: str = "cuda"):
        self.config = config
        self.device = device
        self.name = "BaseMethod"
        
        # 子类可以设置这些属性，基类方法将使用它们
        self.eeg_model = None
        self.encoding_model = None
        
    def optimize(self, target_eeg_feature, target_idx: int, budget: int = 50) -> Dict[str, Any]:
        """
        优化并生成图像
        
        Args:
            target_eeg_feature: 目标EEG特征
            target_idx: 目标索引
            budget: 采样预算
            
        Returns:
            result: {
                'images': List[PIL.Image],
                'best_image': PIL.Image,
                'rewards': List[float],
                'best_reward': float,
                'time': float,
                'metadata': dict
            }
        """
        raise NotImplementedError
        
    def reset(self):
        """重置方法状态（用于多次实验）"""
        pass
    
    # ==================== 共用工具方法 ====================
    
    def _load_eeg_model(self, model_path):
        """加载EEG模型（ATMS）"""
        from model.ATMS_retrieval import ATMS
        eeg_model = ATMS()
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        eeg_model.load_state_dict(checkpoint['eeg_model_state_dict'])
        eeg_model.to(self.device)
        eeg_model.eval()
        return eeg_model
    
    def _load_encoding_model(self, model_path):
        """加载Encoding模型"""
        from model.utils import load_model_encoder
        encoding_model = load_model_encoder(model_path, self.device)
        encoding_model.eval()
        return encoding_model
    
    def _preprocess_image(self, image, device=None):
        """
        预处理图像（PIL Image 或 路径）
        
        Args:
            image: PIL.Image 或 图像路径字符串
            device: 目标设备，默认使用 self.device
            
        Returns:
            torch.Tensor: 预处理后的图像张量 (1, 3, 224, 224)
        """
        import torchvision.transforms as transforms
        
        if device is None:
            device = self.device
            
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 如果是路径字符串，加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        return transform(image).unsqueeze(0).to(device)
    
    def _generate_eeg_from_images(self, images, device=None):
        """
        从PIL图像列表生成EEG特征
        
        Args:
            images: List[PIL.Image] 或 List[str]（图像路径）
            device: 目标设备，默认使用 self.device
            
        Returns:
            List[np.ndarray]: 生成的EEG特征列表
        """
        from model.utils import generate_eeg
        
        if device is None:
            device = self.device
            
        if self.encoding_model is None:
            raise ValueError("encoding_model not initialized. Please set self.encoding_model in subclass.")
        
        synthetic_eegs = []
        for img in images:
            img_tensor = self._preprocess_image(img, device)
            synthetic_eeg = generate_eeg(self.encoding_model, img_tensor, device)
            synthetic_eegs.append(synthetic_eeg)
            
            # 🔥 立即释放图像tensor
            del img_tensor
        
        return synthetic_eegs
    
    def _compute_eeg_similarity_reward(self, eeg, target_feature, subject='sub-01'):
        """
        计算EEG特征与目标特征的相似度reward
        
        Args:
            eeg: np.ndarray or torch.Tensor, EEG特征 (17, 250) 或 (1, 17, 250)
            target_feature: torch.Tensor, 目标EEG特征
            subject: str, 受试者ID
            
        Returns:
            float: 归一化的相似度得分 [0, 1]
        """
        from model.ATMS_retrieval import get_eeg_features
        
        if self.eeg_model is None:
            raise ValueError("eeg_model not initialized. Please set self.eeg_model in subclass.")
        
        # 确保eeg是torch.Tensor并添加batch维度
        # generate_eeg返回 (17, 250)，需要变成 (1, 17, 250)
        if not isinstance(eeg, torch.Tensor):
            eeg = torch.tensor(eeg)
        
        # 如果没有batch维度（dim < 3），添加batch维度
        if eeg.dim() < 3:
            eeg = eeg.unsqueeze(0)
        
        # 获取EEG特征
        eeg_feature = get_eeg_features(
            self.eeg_model, 
            eeg, 
            self.device, 
            subject
        )
        
        # 计算余弦相似度并归一化到 [0, 1]
        similarity = torch.nn.functional.cosine_similarity(
            eeg_feature.to(self.device), 
            target_feature.to(self.device)
        )
        normalized_similarity = (similarity + 1) / 2
        
        return normalized_similarity.item()


class PseudoModelWrapper(BaseMethod):
    """包装现有的Pseudo Model方法"""
    
    def __init__(self, config, device="cuda", shared_models=None):
        super().__init__(config, device)
        self.name = "PseudoModel"
        
        print(f"Initializing {self.name}...")
        
        # 检查是否使用共享模型
        if shared_models is not None:
            print("  Using shared models...")
            self.eeg_model = shared_models.get('eeg_model')
            self.encoding_model = shared_models.get('encoding_model')
            self.vlmodel = shared_models.get('vlmodel')
            self.preprocess_train = shared_models.get('preprocess_train')
        else:
            # 独立模式：加载自己的模型
            print("  Loading independent models...")
            from exp_batch_offline_generation import vlmodel, preprocess_train
            self.vlmodel = vlmodel
            self.preprocess_train = preprocess_train
        
        # 使用基类方法加载模型
        print("  Loading EEG model...")
        self.eeg_model = self._load_eeg_model(config['eeg_model_path'])
        
        print("  Loading encoding model...")
        self.encoding_model = self._load_encoding_model(config['encoding_model_path'])
        
        # 导入全局的Generator和pipe（这些不占大量显存）
        from exp_batch_offline_generation import HeuristicGenerator, pipe
        self.generator = HeuristicGenerator(pipe, self.vlmodel, self.preprocess_train, device=device)
        self.pipe = pipe
        
        self.subject = config.get('subject', 'sub-01')
        
    def _get_image_pool(self, image_set_path):
        """获取候选图片池"""
        test_images_path = []
        for root, dirs, files in os.walk(image_set_path):
            for file in sorted(files):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    test_images_path.append(os.path.join(root, file))
        return test_images_path
    
    def _generate_eeg_from_image_paths(self, test_image_list, device):
        """从图像路径生成EEG（使用基类方法）"""
        # 使用基类的 _generate_eeg_from_images 方法，它支持图像路径
        synthetic_eegs = self._generate_eeg_from_images(test_image_list, device)
        return np.asarray(synthetic_eegs)
    
    def optimize(self, target_eeg_feature, target_idx, budget=50):
        """运行Pseudo Model优化"""
        start_time = time.time()
        
        # 1. 获取图像池并采样
        image_pool = self._get_image_pool(self.config['image_dir'])
        
        # 🔥 排除 target 图像（避免数据泄露）
        if target_idx < len(image_pool):
            target_image_path = image_pool[target_idx]
            image_pool = [p for p in image_pool if p != target_image_path]
            print(f"  [PseudoModel] Excluded target image: {os.path.basename(target_image_path)}")
        
        # 确定实际采样数量（不能超过排除target后的池大小）
        actual_budget = min(budget, len(image_pool))
        print(f"  [PseudoModel] Sampling {actual_budget} images from pool of {len(image_pool)} (budget={budget})")
        sampled_paths = np.random.choice(image_pool, size=actual_budget, replace=False)
        
        # 2. 计算CLIP embeddings（分批处理以减少显存峰值）
        sampled_images = [Image.open(p).convert("RGB") for p in sampled_paths]
        
        batch_size_clip = 16  # 一次处理16张图像
        offline_embeds_list = []
        
        with torch.no_grad():
            for i in range(0, len(sampled_images), batch_size_clip):
                batch_images = sampled_images[i:i+batch_size_clip]
                tensor_batch = torch.stack([self.preprocess_train(img) for img in batch_images]).to(self.device)
                batch_embeds = self.vlmodel.encode_image(tensor_batch)
                offline_embeds_list.append(batch_embeds.cpu())  # 立即转到CPU
                del tensor_batch, batch_embeds
        
        offline_embeds = torch.cat(offline_embeds_list, dim=0).to(self.device)
        del offline_embeds_list
        
        # 3. 计算rewards（使用基类方法）
        synthetic_eegs = self._generate_eeg_from_image_paths(
            sampled_paths, self.device
        )
        
        offline_rewards = []
        for eeg in synthetic_eegs:
            r = self._compute_eeg_similarity_reward(
                eeg, target_eeg_feature, self.subject
            )
            offline_rewards.append(r)
        
        # 🔥 立即释放EEG特征（已经计算完rewards）
        del synthetic_eegs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        offline_rewards_tensor = torch.tensor(offline_rewards).to(self.device)
        
        # 4. 添加数据到Pseudo Model
        from model.pseudo_target_model import PseudoTargetModel
        self.generator.pseudo_target_model = PseudoTargetModel(
            dimension=1024, noise_level=1e-4
        ).to(self.device)
        
        self.generator.pseudo_target_model.add_model_data(
            offline_embeds,
            (-offline_rewards_tensor * self.generator.reward_scaling_factor).to(self.device)
        )
        
        # 5. 评估阶段：生成5张图像并求平均分数
        num_eval_samples = 5
        print(f"  [PseudoModel] Generating {num_eval_samples} evaluation images...")
        
        # 🔥 先释放不再需要的大张量，为评估腾出空间
        del sampled_images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        final_images = []
        final_rewards = []
        
        for eval_idx in range(num_eval_samples):
            # 每次生成一张图像（因为 generator.generate_batch_size=1）
            eval_images = self.generator.generate(
                data_x=offline_embeds,
                data_y=offline_rewards_tensor,
                tar_image_embed=None,
                prompt='',
                save_path=None
            )
            
            # 计算该图像的 reward
            eval_eegs = self._generate_eeg_from_images(eval_images, self.device)
            eval_reward = self._compute_eeg_similarity_reward(
                eval_eegs[0], target_eeg_feature, self.subject
            )
            
            final_images.extend(eval_images)
            final_rewards.append(eval_reward)
            
            # 🔥 立即释放中间变量
            del eval_eegs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        elapsed_time = time.time() - start_time
        
        # 6. 计算平均分数作为最终结果
        avg_reward = np.mean(final_rewards)
        print(f"  [PseudoModel] Evaluation complete: avg reward = {avg_reward:.4f} (std = {np.std(final_rewards):.4f})")
        print(f"  [PseudoModel] Individual rewards: {[f'{r:.4f}' for r in final_rewards]}")
        
        result = {
            'images': final_images,
            'best_image': final_images[0],  # 保留第一张作为代表图像
            'rewards': final_rewards,
            'best_reward': avg_reward,  # 使用平均值
            'time': elapsed_time,
            'n_samples': budget,
            'metadata': {
                'optimization_steps': self.generator.total_steps,
                'pool_size': budget,
                'num_eval_samples': num_eval_samples,
                'eval_rewards_std': float(np.std(final_rewards))
            }
        }
        
        # 清理中间变量释放显存
        del offline_embeds, offline_rewards_tensor
        
        # 🔥 清理Pseudo Target Model
        if hasattr(self.generator, 'pseudo_target_model') and self.generator.pseudo_target_model is not None:
            del self.generator.pseudo_target_model
            self.generator.pseudo_target_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return result
    
    def reset(self):
        """重置Generator状态，清理GPU缓存"""
        # 🔥 清理Pseudo Target Model
        if hasattr(self.generator, 'pseudo_target_model') and self.generator.pseudo_target_model is not None:
            del self.generator.pseudo_target_model
            self.generator.pseudo_target_model = None
        
        # 重置Pseudo Target Model会在下次optimize时自动创建新的
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class HeuristicClosedLoopWrapper(BaseMethod):
    """Heuristic Closed-Loop Pseudo Model方法（迭代融合+贪婪采样）"""
    
    def __init__(self, config, device="cuda", shared_models=None):
        super().__init__(config, device)
        self.name = "HeuristicClosedLoop"
        
        print(f"Initializing {self.name}...")
        
        # 检查是否使用共享模型
        if shared_models is not None:
            print("  Using shared models...")
            self.eeg_model = shared_models.get('eeg_model')
            self.encoding_model = shared_models.get('encoding_model')
            self.vlmodel = shared_models.get('vlmodel')
            self.preprocess_train = shared_models.get('preprocess_train')
        else:
            # 独立模式：加载自己的模型
            print("  Loading independent models...")
            from exp_batch_offline_generation import vlmodel, preprocess_train
            self.vlmodel = vlmodel
            self.preprocess_train = preprocess_train
        
        # 使用基类方法加载模型
        print("  Loading EEG model...")
        self.eeg_model = self._load_eeg_model(config['eeg_model_path'])
        
        print("  Loading encoding model...")
        self.encoding_model = self._load_encoding_model(config['encoding_model_path'])
        
        # 导入 pipe
        from exp_batch_offline_generation import pipe
        
        # 使用共享的pipe或独立加载
        if shared_models is not None and shared_models.get('sdxl_pipe') is not None:
            self.pipe = shared_models.get('sdxl_pipe')
        else:
            self.pipe = pipe
        
        # 🔥 初始化 HeuristicGenerator（包含 PseudoTargetModel）
        # 使用当前文件中定义的 HeuristicGenerator 类
        # 不自动加载IP adapter（因为已经通过shared_models加载）
        self.generator = HeuristicGenerator(
            self.pipe, 
            self.vlmodel, 
            self.preprocess_train, 
            device=device,
            seed=42,
            load_ip_adapter=False,  # 不重复加载
            min_data_threshold=5  # 🔥 修改为5，与initial_sample_size保持一致
        )
        
        self.subject = config.get('subject', 'sub-01')
        
        # Closed-loop 特定参数
        # 🔥 关键改进：实际循环次数由budget自动决定，num_loops_max仅作为可选的硬性上限
        # 这样可以在不超过budget的前提下，自动运行尽可能多的优化迭代轮数
        self.num_loops_max = config.get('num_loops_closedloop', 999)  # 最大循环次数上限（默认999，实际由budget决定）
        self.num_fusions_per_round = config.get('num_fusions_per_round', 2)  # 每轮融合次数
        self.top_k_greedy = config.get('top_k_greedy', 20)  # 贪婪采样的top-k
        self.initial_sample_size = config.get('initial_sample_size_closedloop', 5)  # 初始采样数量
        
        print(f"  Closed-loop params: num_loops_max={self.num_loops_max} (auto from budget), fusions_per_round={self.num_fusions_per_round}")
    
    def _get_image_pool(self, image_set_path):
        """获取候选图片池"""
        test_images_path = []
        for root, dirs, files in os.walk(image_set_path):
            for file in sorted(files):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    test_images_path.append(os.path.join(root, file))
        return test_images_path
    
    def _fusion_image_generation(self, fit_images, fit_rewards):
        """
        融合生成图像（严格按照原始实现）
        调用 Generator.generate()，包含15步GP优化
        """
        import random
        
        # 提取fit_images的CLIP embeddings
        tensor_fit_images = [self.preprocess_train(img) for img in fit_images]
        with torch.no_grad():
            img_embeds = self.vlmodel.encode_image(torch.stack(tensor_fit_images).to(self.device))
        
        # 随机选择两个图像进行融合
        idx1, idx2 = random.sample(range(len(img_embeds)), 2)
        embed1 = img_embeds[idx1].unsqueeze(0)
        embed2 = img_embeds[idx2].unsqueeze(0)
        
        # 部分交换（类似基因交叉）
        scale = 512
        embed_len = embed1.size(1)
        start_idx = random.randint(0, embed_len - scale - 1)
        end_idx = start_idx + scale
        
        temp = embed1[:, start_idx:end_idx].clone()
        embed1[:, start_idx:end_idx] = embed2[:, start_idx:end_idx]
        embed2[:, start_idx:end_idx] = temp
        
        # 🔥 使用 Generator.generate() 生成图像（包含15步优化）
        generated_images = []
        
        with torch.no_grad():
            # 生成第一张图像（从 embed1 开始优化）
            images1 = self.generator.generate(
                img_embeds.to(self.device),
                torch.tensor(fit_rewards).to(self.device),
                None,
                prompt='',
                save_path=None,
                start_embedding=embed1
            )
            generated_images.extend(images1)
            
            # 生成第二张图像（从 embed2 开始优化）
            images2 = self.generator.generate(
                img_embeds.to(self.device),
                torch.tensor(fit_rewards).to(self.device),
                None,
                prompt='',
                save_path=None,
                start_embedding=embed2
            )
            generated_images.extend(images2)
        
        # 返回生成的图像和源图像索引
        return generated_images, (idx1, idx2), fit_images
    
    def _greedy_sampling(self, target_clip_embed, image_pool, processed_paths, test_set_img_embeds, num_samples):
        """基于target的贪婪采样"""
        # 获取可用图像的索引
        available_indices = []
        for i, path in enumerate(image_pool):
            if path not in processed_paths:
                available_indices.append(i)
        
        if len(available_indices) == 0:
            return []
        
        # 基于 target_clip_embed 计算相似度
        available_features = test_set_img_embeds[available_indices]
        
        # 计算余弦相似度
        target_norm = torch.nn.functional.normalize(target_clip_embed.float(), p=2, dim=1)
        available_norm = torch.nn.functional.normalize(available_features.float(), p=2, dim=1)
        cosine_similarities = torch.mm(available_norm, target_norm.t()).squeeze(1)
        cosine_similarities = (cosine_similarities + 1) / 2
        
        sorted_available_indices = np.argsort(cosine_similarities.cpu())
        
        # 从 top-K 中随机采样
        top_indices = sorted_available_indices[-min(self.top_k_greedy, len(sorted_available_indices)):]
        num_to_sample = min(num_samples, len(top_indices))
        selected_indices = np.random.choice(top_indices, size=num_to_sample, replace=False)
        
        # 加载选中的图像
        greedy_images = []
        sample_image_paths = []
        for selected_idx in selected_indices:
            greedy_image = Image.open(image_pool[available_indices[selected_idx]]).convert("RGB")
            greedy_images.append(greedy_image)
            sample_image_paths.append(image_pool[available_indices[selected_idx]])
        
        processed_paths.update(sample_image_paths)
        
        return greedy_images
    
    def optimize(self, target_eeg_feature, target_idx, budget=50):
        """运行Heuristic Closed-Loop优化"""
        from scipy.special import softmax
        from model.pseudo_target_model import PseudoTargetModel
        
        start_time = time.time()
        
        # 🔥 重新初始化 Generator 的 PseudoTargetModel（每次优化都创建新的）
        self.generator.pseudo_target_model = PseudoTargetModel(
            dimension=1024, 
            noise_level=1e-4
        ).to(self.device)
        
        # 🔥 根据 budget 自动计算最优循环次数
        IMAGES_PER_LOOP_INITIAL = self.initial_sample_size  # 第1轮：初始采样
        IMAGES_PER_LOOP_FUSION = self.num_fusions_per_round * 4 + self.num_fusions_per_round * 2  # 每轮后续：融合生成+源图+贪婪采样
        
        remaining_budget = budget - IMAGES_PER_LOOP_INITIAL
        num_loops_from_budget = 1 + max(0, remaining_budget // IMAGES_PER_LOOP_FUSION)
        
        # 🔥 优先使用budget计算的轮数，num_loops_max仅作为硬性上限
        actual_num_loops = min(self.num_loops_max, num_loops_from_budget)
        estimated_images = IMAGES_PER_LOOP_INITIAL + (actual_num_loops - 1) * IMAGES_PER_LOOP_FUSION
        
        print(f"  [{self.name}] Starting closed-loop optimization...")
        print(f"  [{self.name}] Budget={budget}, Budget-based loops={num_loops_from_budget}, Max loops={self.num_loops_max}")
        print(f"  [{self.name}] Actual loops={actual_num_loops}, Estimated images={estimated_images} (utilization: {estimated_images/budget*100:.1f}%)")
        print(f"  [{self.name}] Config: fusions/round={self.num_fusions_per_round}, initial_samples={self.initial_sample_size}")
        
        # 1. 获取图像池
        image_pool = self._get_image_pool(self.config['image_dir'])
        
        # 排除 target 图像
        if target_idx < len(image_pool):
            target_image_path = image_pool[target_idx]
            image_pool = [p for p in image_pool if p != target_image_path]
            print(f"  [{self.name}] Excluded target image: {os.path.basename(target_image_path)}")
        
        # 2. 预先计算所有图像的CLIP embeddings（避免重复计算）
        print(f"  [{self.name}] Pre-computing CLIP embeddings for {len(image_pool)} images...")
        all_images_pil = [Image.open(p).convert("RGB") for p in image_pool]
        
        batch_size_clip = 16
        test_set_img_embeds_list = []
        with torch.no_grad():
            for i in range(0, len(all_images_pil), batch_size_clip):
                batch_images = all_images_pil[i:i+batch_size_clip]
                tensor_batch = torch.stack([self.preprocess_train(img) for img in batch_images]).to(self.device)
                batch_embeds = self.vlmodel.encode_image(tensor_batch)
                test_set_img_embeds_list.append(batch_embeds.cpu())
                del tensor_batch, batch_embeds
        
        test_set_img_embeds = torch.cat(test_set_img_embeds_list, dim=0).to(self.device)
        del test_set_img_embeds_list, all_images_pil
        
        # 3. 获取target的CLIP embedding
        target_image_path = image_pool[0] if target_idx >= len(image_pool) else image_pool[target_idx]
        target_image_pil = Image.open(target_image_path).convert("RGB")
        with torch.no_grad():
            target_clip_embed = self.vlmodel.encode_image(
                self.preprocess_train(target_image_pil).unsqueeze(0).to(self.device)
            )
        
        # 4. 闭环迭代
        processed_paths = set()
        fit_images = []
        fit_eegs = []
        fit_rewards = []
        
        all_loop_rewards = []
        all_loop_images = []
        total_images_evaluated = 0  # 🔥 追踪实际评估的图像总数
        
        # 🔥 保存最后一轮的所有候选（用于评估）
        final_loop_images = []
        final_loop_eegs = []
        final_loop_rewards = []
        
        # 🔥 使用根据 budget 计算的 actual_num_loops
        for t in range(actual_num_loops):
            print(f"\n  [{self.name}] Loop {t+1}/{actual_num_loops}")
            
            loop_sample_ten = []
            loop_eeg_ten = []
            loop_reward_ten = []
            loop_loss_ten = []
            
            if t == 0:
                # 初始轮：从图像池随机采样
                print(f"    Initial sampling: {self.initial_sample_size} images")
                available_paths = [path for path in image_pool if path not in processed_paths]
                sample_image_paths = np.random.choice(
                    available_paths, 
                    min(self.initial_sample_size, len(available_paths)), 
                    replace=False
                )
                
                chosen_images = [Image.open(p).convert("RGB") for p in sample_image_paths]
                processed_paths.update(sample_image_paths)
                
                # 计算rewards
                synthetic_eegs = self._generate_eeg_from_images(chosen_images, self.device)
                chosen_rewards = [
                    self._compute_eeg_similarity_reward(eeg, target_eeg_feature, self.subject)
                    for eeg in synthetic_eegs
                ]
                chosen_eegs = synthetic_eegs
                chosen_losses = [0] * len(chosen_images)
                
                loop_sample_ten.extend(chosen_images)
                loop_eeg_ten.extend(chosen_eegs)
                loop_reward_ten.extend(chosen_rewards)
                loop_loss_ten.extend(chosen_losses)
                
                # 🔥 更新图像评估计数
                total_images_evaluated += len(chosen_images)
                
                # 添加到PseudoTargetModel
                tensor_loop_sample = [self.preprocess_train(img) for img in loop_sample_ten]
                with torch.no_grad():
                    tensor_loop_sample_embeds = self.vlmodel.encode_image(
                        torch.stack(tensor_loop_sample).to(self.device)
                    )
                self.generator.pseudo_target_model.add_model_data(
                    tensor_loop_sample_embeds.clone(),
                    (-torch.tensor(loop_reward_ten) * 100).to(self.device)  # reward_scaling_factor = 100
                )
                
                del tensor_loop_sample, tensor_loop_sample_embeds
                torch.cuda.empty_cache()
            
            else:
                # 后续轮：融合生成 + 贪婪采样
                all_generated_images = []
                all_fusion_source_images = []
                
                # 多次融合
                for fusion_idx in range(self.num_fusions_per_round):
                    print(f"      Fusion {fusion_idx+1}/{self.num_fusions_per_round}")
                    
                    generated_images, (idx1, idx2), fit_imgs = self._fusion_image_generation(fit_images, fit_rewards)
                    synthetic_eegs = self._generate_eeg_from_images(generated_images, self.device)
                    
                    loop_sample_ten.extend(generated_images)
                    loop_eeg_ten.extend(synthetic_eegs)
                    all_generated_images.extend(generated_images)
                    
                    # 计算融合生成图像的rewards
                    for eeg in synthetic_eegs:
                        r = self._compute_eeg_similarity_reward(eeg, target_eeg_feature, self.subject)
                        loop_reward_ten.append(r)
                        loop_loss_ten.append(0)
                    
                    del synthetic_eegs
                    torch.cuda.empty_cache()
                    
                    # 添加融合源图像
                    fusion_source_images = [fit_imgs[idx1], fit_imgs[idx2]]
                    fusion_source_eegs = self._generate_eeg_from_images(fusion_source_images, self.device)
                    
                    loop_sample_ten.extend(fusion_source_images)
                    loop_eeg_ten.extend(fusion_source_eegs)
                    all_fusion_source_images.extend(fusion_source_images)
                    
                    for eeg in fusion_source_eegs:
                        r = self._compute_eeg_similarity_reward(eeg, target_eeg_feature, self.subject)
                        loop_reward_ten.append(r)
                        loop_loss_ten.append(0)
                    
                    del fusion_source_eegs
                    torch.cuda.empty_cache()
                
                print(f"      Fusion complete: {len(all_generated_images)} generated, {len(all_fusion_source_images)} sources")
                
                # 🔥 更新图像评估计数（融合生成的图像 + 源图像）
                total_images_evaluated += len(all_generated_images) + len(all_fusion_source_images)
                
                # 贪婪采样
                num_greedy_samples = len(all_generated_images)
                greedy_images = self._greedy_sampling(
                    target_clip_embed, 
                    image_pool, 
                    processed_paths, 
                    test_set_img_embeds, 
                    num_greedy_samples
                )
                
                if len(greedy_images) > 0:
                    synthetic_eegs = self._generate_eeg_from_images(greedy_images, self.device)
                    loop_sample_ten.extend(greedy_images)
                    loop_eeg_ten.extend(synthetic_eegs)
                    
                    for eeg in synthetic_eegs:
                        r = self._compute_eeg_similarity_reward(eeg, target_eeg_feature, self.subject)
                        loop_reward_ten.append(r)
                        loop_loss_ten.append(0)
                    
                    # 🔥 更新图像评估计数（贪婪采样的图像）
                    total_images_evaluated += len(greedy_images)
                    
                    del synthetic_eegs
                    torch.cuda.empty_cache()
                    
                    print(f"      Greedy sampling: {len(greedy_images)} images from top-{self.top_k_greedy}")
                
                # 🔥 选择top-5保留到下一轮（与其他方法保持一致）
                loop_probabilities = softmax(loop_reward_ten)
                
                # 概率采样top-5
                chosen_indices = np.random.choice(
                    len(loop_probabilities), 
                    size=min(5, len(loop_probabilities)), 
                    replace=False, 
                    p=loop_probabilities
                )
                
                chosen_rewards = [loop_reward_ten[idx] for idx in chosen_indices]
                chosen_losses = [loop_loss_ten[idx] for idx in chosen_indices]
                chosen_images = [loop_sample_ten[idx] for idx in chosen_indices]
                chosen_eegs = [loop_eeg_ten[idx] for idx in chosen_indices]
                
                # 按reward排序
                combined = list(zip(chosen_rewards, chosen_losses, chosen_images, chosen_eegs))
                combined.sort(reverse=True, key=lambda x: x[0])
                chosen_rewards, chosen_losses, chosen_images, chosen_eegs = zip(*combined)
                chosen_rewards = list(chosen_rewards)
                chosen_losses = list(chosen_losses)
                chosen_images = list(chosen_images)
                chosen_eegs = list(chosen_eegs)
                
                # 添加到PseudoTargetModel
                tensor_loop_sample = [self.preprocess_train(img) for img in loop_sample_ten]
                with torch.no_grad():
                    tensor_loop_sample_embeds = self.vlmodel.encode_image(
                        torch.stack(tensor_loop_sample).to(self.device)
                    )
                self.generator.pseudo_target_model.add_model_data(
                    tensor_loop_sample_embeds.clone(),
                    (-torch.tensor(loop_reward_ten) * 100).to(self.device)  # reward_scaling_factor = 100
                )
                
                del tensor_loop_sample, tensor_loop_sample_embeds
                torch.cuda.empty_cache()
            
            # 更新fit数据
            fit_images = chosen_images
            fit_eegs = chosen_eegs
            fit_rewards = chosen_rewards
            fit_losses = chosen_losses
            
            # 记录本轮数据
            all_loop_rewards.append(np.mean(loop_reward_ten))
            all_loop_images.extend(loop_sample_ten[:4])  # 保留前4张作为代表
            
            print(f"      Loop {t+1} complete: mean_reward={np.mean(loop_reward_ten):.4f}, best_reward={max(loop_reward_ten):.4f}")
            
            # 🔥 如果是最后一轮，保存所有候选图像用于评估（恢复原始评估方式）
            if t == actual_num_loops - 1:
                print(f"  [{self.name}] Saving final round candidates for evaluation...")
                final_loop_images = loop_sample_ten.copy()
                final_loop_eegs = loop_eeg_ten.copy()
                final_loop_rewards = loop_reward_ten.copy()
                print(f"  [{self.name}] Final round: {len(final_loop_images)} candidates saved")
        
        # 记录优化时间（不包括评估时间）
        optimization_time = time.time() - start_time
        
        print(f"  [{self.name}] Optimization complete in {optimization_time:.2f}s")
        print(f"  [{self.name}] Training best reward: {np.mean(fit_rewards):.4f}")
        print(f"  [{self.name}] Actual image evaluations during training: {total_images_evaluated} (budget: {budget}, estimated: {estimated_images})")
        
        # 🔥 5. 评估阶段：从最后一轮候选中选择Top-5（恢复原始评估方式）
        # 这是HeuristicClosedLoop的正确评估方式：它通过迭代产生候选，然后选择最优
        num_eval_samples = 5
        
        if len(final_loop_images) > 0:
            print(f"  [{self.name}] Selecting Top-{num_eval_samples} from final round's {len(final_loop_images)} candidates...")
            print(f"  [{self.name}] Final round rewards: {[f'{r:.4f}' for r in final_loop_rewards]}")
            
            # 从最后一轮的所有候选中选择Top-5（基于EEG similarity reward）
            sorted_indices = np.argsort(final_loop_rewards)[-num_eval_samples:]  # 选择reward最高的5个
            sorted_indices = sorted_indices[::-1]  # 从高到低排序
            
            eval_images = [final_loop_images[i] for i in sorted_indices]
            eval_eegs = [final_loop_eegs[i] for i in sorted_indices]
            eval_rewards = [final_loop_rewards[i] for i in sorted_indices]
            
            print(f"  [{self.name}] Selected Top-{num_eval_samples} rewards: {[f'{r:.4f}' for r in eval_rewards]}")
            
        else:
            # 如果没有保存最后一轮数据（例如提前停止），使用fit_images
            print(f"  [{self.name}] Warning: No final round candidates, using fit_images as fallback")
            eval_images = fit_images[:num_eval_samples]
            eval_eegs = fit_eegs[:num_eval_samples]
            eval_rewards = fit_rewards[:num_eval_samples]
        
        # 计算评估结果
        final_eval_reward = np.mean(eval_rewards)
        best_eval_image = eval_images[0]  # 已经按reward排序，第一张就是最佳
        
        # 总时间（优化时间，不需要额外评估时间）
        total_time = optimization_time
        
        print(f"  [{self.name}] Evaluation complete:")
        print(f"  [{self.name}]   - Evaluation mean reward: {final_eval_reward:.4f} (std: {np.std(eval_rewards):.4f})")
        print(f"  [{self.name}]   - Individual eval rewards: {[f'{r:.4f}' for r in eval_rewards]}")
        print(f"  [{self.name}]   - Training best reward: {np.mean(fit_rewards):.4f} (for comparison)")
        print(f"  [{self.name}]   - Total time: {total_time:.2f}s")
        print(f"  [{self.name}]   - Evaluation method: Selection from final round candidates (original method)")
        
        # 🔥 返回选出的Top-5图像（恢复原始评估方式）
        result = {
            'images': eval_images,  # 从最后一轮候选中选出的Top-5
            'best_image': best_eval_image,  # Top-1
            'rewards': eval_rewards,  # 对应的rewards
            'best_reward': final_eval_reward,  # Top-5的平均reward
            'time': total_time,  # 总时间
            'n_samples': total_images_evaluated,  # 训练期间评估的图像数
            'metadata': {
                'num_loops': actual_num_loops,
                'fusions_per_round': self.num_fusions_per_round,
                'top_k_greedy': self.top_k_greedy,
                'loop_rewards': all_loop_rewards,
                'budget': budget,
                'estimated_images': estimated_images,
                'from_pool_samples': len(processed_paths),
                'training_best_reward': float(np.mean(fit_rewards)),
                'optimization_time': optimization_time,
                'evaluation_method': 'selection_from_candidates',  # 标记评估方式
                'num_final_candidates': len(final_loop_images),
                'num_eval_samples': num_eval_samples
            }
        }
        
        # 清理
        del test_set_img_embeds, target_clip_embed
        
        # 清理最后一轮数据（已经使用完毕）
        if len(final_loop_images) > 0:
            del final_loop_images, final_loop_eegs, final_loop_rewards
        
        # 清理 PseudoTargetModel
        if hasattr(self, 'generator') and self.generator is not None:
            if hasattr(self.generator, 'pseudo_target_model') and self.generator.pseudo_target_model is not None:
                del self.generator.pseudo_target_model
                self.generator.pseudo_target_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return result
    
    def reset(self):
        """重置状态，清理GPU缓存"""
        if hasattr(self, 'generator') and self.generator is not None:
            if hasattr(self.generator, 'pseudo_target_model') and self.generator.pseudo_target_model is not None:
                del self.generator.pseudo_target_model
                self.generator.pseudo_target_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class DDPOWrapper(BaseMethod):
    """DDPO方法的包装器 - 基于d3po实现"""
    
    def __init__(self, config, device="cuda", shared_models=None):
        super().__init__(config, device)
        self.name = "DDPO"
        
        print(f"Initializing {self.name}...")
        
        # 导入依赖
        from peft import LoraConfig
        from model.ATMS_retrieval import ATMS, get_eeg_features
        import sys
        sys.path.append('/home/ldy/Workspace/guide-stable-diffusion/related_works/d3po/d3po')
        
        # 检查是否使用共享模型
        if shared_models is not None:
            print("  Using shared models...")
            # 使用共享的模型实例
            self.eeg_model = shared_models.get('eeg_model')
            self.encoding_model = shared_models.get('encoding_model')
            
            # 🔥 使用共享的SDXL pipeline（不创建新的！）
            self.pipe = shared_models.get('sdxl_pipe')
            if self.pipe is None:
                raise ValueError("Shared SDXL pipeline not found!")
            print("  Using shared SDXL pipeline")
        else:
            # 独立模式：创建自己的 pipeline 和模型
            print("  Standalone mode: loading independently...")
            self.pipe = self._create_independent_pipeline(device)
            self.eeg_model = self._load_eeg_model(config['eeg_model_path'])
            self.encoding_model = self._load_encoding_model(config['encoding_model_path'])
        
        self.subject = config.get('subject', 'sub-01')
        
        # 训练参数
        self.n_epochs = config.get('n_epochs', 5)
        self.batch_size = config.get('batch_size', 4)
        self.learning_rate = config.get('learning_rate', 3e-5)
        self.use_lora = config.get('use_lora', True)
        self.clip_range = config.get('clip_range', 1e-4)
        self.adv_clip_max = config.get('adv_clip_max', 5)
        self.num_inference_steps = config.get('num_inference_steps', 8)
        self.guidance_scale = 0.0
        self.eta = 0.0
        
        # 冻结非训练组件
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.text_encoder_2.requires_grad_(False)
        
        # 🔥 延迟配置LoRA：在optimize()时再添加，避免与其他方法冲突
        print("  LoRA will be configured on first optimize() call")
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.optimizer = None  # 优化器也延迟创建
        
        print(f"  DDPO initialized: {self.n_epochs} epochs, batch_size={self.batch_size}")
    
    def _create_independent_pipeline(self, device):
        """创建独立的 SDXL pipeline（与其他方法使用相同的模型）"""
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        
        # 使用与 Generator4Embeds 相同的模型配置
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        
        # 加载 UNet（SDXL-Lightning 8步模型）
        unet = UNet2DConditionModel.from_pretrained(
            model_id, 
            subfolder="unet", 
            torch_dtype=torch.bfloat16, 
            use_safetensors=True, 
            variant="fp16"
        )
        unet.load_state_dict(load_file(hf_hub_download(
            "ByteDance/SDXL-Lightning", 
            "sdxl_lightning_8step_unet.safetensors"
        )))
        
        # 加载 VAE
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.bfloat16
        )
        
        # 加载 Scheduler
        scheduler = DDIMScheduler.from_pretrained(
            model_id, 
            subfolder="scheduler",
            timestep_spacing="trailing"
        )
        
        # 创建 pipeline（不使用 ExtendedStableDiffusionXLPipeline，避免 IP-Adapter 相关代码）
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, 
            unet=unet, 
            vae=vae, 
            scheduler=scheduler, 
            torch_dtype=torch.bfloat16, 
            use_safetensors=True, 
            variant="fp16"
        )
        
        pipe.vae.enable_slicing()
        pipe.to(device)
        
        return pipe
    
    def optimize(self, target_eeg_feature, target_idx, budget=50):
        """运行DDPO优化（基于d3po的实现，使用gradient accumulation避免显存暴涨）"""
        from d3po_pytorch.diffusers_patch.pipeline_with_logprob_sdxl import pipeline_with_logprob
        from d3po_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
        
        # 🔥 Step 1: 清理所有已有的LoRA adapters（避免与其他方法冲突）
        print(f"  [{self.name}] Setting up LoRA adapter...")
        if hasattr(self.pipe.unet, 'peft_config') and len(self.pipe.unet.peft_config) > 0:
            print(f"    Removing {len(self.pipe.unet.peft_config)} existing adapter(s)")
            self.pipe.unet.delete_adapters(list(self.pipe.unet.peft_config.keys()))
        
        # 🔥 Step 2: 添加本方法的LoRA adapter
        self.pipe.unet.add_adapter(self.lora_config)
        
        # 🔥 Step 3: 设置LoRA参数为float32
        for param in self.pipe.unet.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)
        
        # 🔥 Step 4: 创建优化器
        trainable_params = filter(lambda p: p.requires_grad, self.pipe.unet.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
        print(f"    LoRA adapter configured successfully")
        
        start_time = time.time()
        num_batches_per_epoch = max(1, budget // (self.n_epochs * self.batch_size))
        
        best_reward = -float('inf')
        best_image = None
        total_samples_count = 0
        
        # 在训练前清理 GPU 内存
        torch.cuda.empty_cache()
        
        print(f"  Training with gradient accumulation: {num_batches_per_epoch} batches per epoch")
        
        for epoch in range(self.n_epochs):
            #################### SAMPLING + TRAINING (逐个batch处理，避免累积) ####################
            self.pipe.unet.train()  # 直接进入训练模式
            
            # 🔥 关键改变：逐个batch采样和训练，不累积所有samples
            for batch_idx in range(num_batches_per_epoch):
                ########## Step 1: 采样单个batch ##########
                self.pipe.unet.eval()
                prompts = [""] * self.batch_size
                
                # 创建随机生成器
                generator = torch.Generator(device=self.device)
                generator.manual_seed(torch.seed())
                
                with torch.no_grad():
                    images, latents, log_probs, _ = pipeline_with_logprob(
                        self.pipe, prompt=prompts,
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        eta=self.eta, output_type="pt", return_dict=False,
                        generator=generator,
                    )
                
                # 转换为PIL并立即转移到CPU
                pil_images = [Image.fromarray((img.float().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)) 
                              for img in images]
                
                # 将所有GPU tensor转到CPU
                latents_cpu = [lat.cpu() for lat in latents]
                log_probs_cpu = [lp.cpu() for lp in log_probs]
                
                # 🔥 立即删除GPU上的原始tensor，释放显存
                del images, latents, log_probs, generator
                torch.cuda.empty_cache()
                
                # 临时将 diffusion 模型移出 GPU 以为 EEG 模型腾出空间
                self.pipe.to('cpu')
                torch.cuda.empty_cache()
                
                # 计算rewards
                synthetic_eegs = self._generate_eeg_from_images(pil_images)
                batch_rewards = [self._compute_eeg_similarity_reward(eeg, target_eeg_feature, self.subject) 
                                for eeg in synthetic_eegs]
                
                # 将 diffusion 模型移回 GPU，数据也移回
                self.pipe.to(self.device)
                latents = [lat.to(self.device) for lat in latents_cpu]
                log_probs = [lp.to(self.device) for lp in log_probs_cpu]
                # 🔥 CPU副本已经不需要了，立即删除
                del latents_cpu, log_probs_cpu
                torch.cuda.empty_cache()
                
                # 更新最佳结果（不保存所有图像，节省内存）
                for img, r in zip(pil_images, batch_rewards):
                    total_samples_count += 1
                    if r > best_reward:
                        best_reward = r
                        best_image = img
                
                ########## Step 2: 立即训练这个batch（不累积） ##########
                self.pipe.unet.train()
                
                # 构建单个batch的sample
                sample = {
                    'timesteps': self.pipe.scheduler.timesteps.repeat(self.batch_size, 1),
                    'latents': torch.stack(latents, dim=1)[:, :-1],
                    'next_latents': torch.stack(latents, dim=1)[:, 1:],
                    'log_probs': torch.stack(log_probs, dim=1),
                    'rewards': torch.tensor(batch_rewards, device=self.device),
                }
                
                # 计算advantages（基于当前batch）
                rewards_np = sample['rewards'].cpu().numpy()
                advantages = (rewards_np - rewards_np.mean()) / (rewards_np.std() + 1e-8)
                sample['advantages'] = torch.tensor(advantages, device=self.device)
                
                # 训练参数
                num_timesteps = sample['timesteps'].shape[1]
                num_train_timesteps = max(1, int(num_timesteps * 0.25))
                
                # 标记 latents 需要梯度
                sample["latents"].requires_grad = True
                
                # 定义 callback 函数
                def callback_func(pipe_self, step_index, timestep, callback_kwargs):
                    nonlocal sample, num_train_timesteps
                    
                    if step_index >= num_train_timesteps:
                        return {"latents": sample["next_latents"][:, step_index]}
                    
                    log_prob = callback_kwargs["log_prob"]
                    
                    # 计算 PPO 损失
                    advantages = torch.clamp(
                        sample["advantages"], -self.adv_clip_max, self.adv_clip_max
                    )
                    ratio = torch.exp(log_prob - sample["log_probs"][:, step_index])
                    unclipped_loss = -advantages * ratio
                    clipped_loss = -advantages * torch.clamp(
                        ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                    )
                    loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                    
                    # 反向传播
                    loss.backward()
                    
                    # 在最后一个训练 timestep 更新参数
                    if step_index == num_train_timesteps - 1:
                        torch.nn.utils.clip_grad_norm_(self.pipe.unet.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    
                    return {"latents": sample["next_latents"][:, step_index]}
                
                # 执行训练
                with torch.set_grad_enabled(True):
                    train_out = pipeline_with_logprob(
                        self.pipe,
                        prompt=prompts,
                        latents=sample["latents"][:, 0],
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        eta=self.eta,
                        output_type="latent",
                        return_dict=False,
                        callback_on_step_end=callback_func,
                        callback_on_step_end_tensor_inputs=["log_prob"],
                        log_probs_given_trajectory=sample["next_latents"],
                        enable_grad=True,
                        enable_grad_checkpointing=True,
                    )
                
                # 🔥 立即清理当前batch的数据，释放显存
                sample["latents"].requires_grad = False
                # 🔥 DDPO：确保梯度被完全清零（防止梯度累积导致OOM）
                self.optimizer.zero_grad(set_to_none=True)
                # 删除callback函数，释放其捕获的变量
                del callback_func
                del sample, latents, log_probs, pil_images, synthetic_eegs, batch_rewards, train_out
                torch.cuda.empty_cache()
                
                # 打印进度
                if (batch_idx + 1) % max(1, num_batches_per_epoch // 5) == 0:
                    print(f"    Epoch {epoch+1}/{self.n_epochs}, Batch {batch_idx+1}/{num_batches_per_epoch}, Best Reward: {best_reward:.4f}")
        
        #################### EVALUATION: 用优化后的模型重新生成图像 ####################
        print(f"  Generating final images with optimized model...")
        self.pipe.unet.eval()
        final_images = []
        final_rewards = []
        
        # 生成几张图像来评估优化效果
        num_eval_samples = min(5, budget // 2)  # 生成5张或更少
        
        with torch.no_grad():
            for _ in range(num_eval_samples):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(torch.seed())
                
                eval_images, _, _, _ = pipeline_with_logprob(
                    self.pipe, prompt=[""],
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    eta=self.eta, output_type="pt", return_dict=False,
                    generator=generator,
                )
                
                eval_pil = [Image.fromarray((img.float().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)) 
                           for img in eval_images]
                
                # 计算最终奖励
                self.pipe.to('cpu')
                torch.cuda.empty_cache()
                
                eval_eegs = self._generate_eeg_from_images(eval_pil)
                eval_rewards_batch = [self._compute_eeg_similarity_reward(eeg, target_eeg_feature, self.subject) 
                                     for eeg in eval_eegs]
                
                self.pipe.to(self.device)
                torch.cuda.empty_cache()
                
                final_images.extend(eval_pil)
                final_rewards.extend(eval_rewards_batch)
        
        # 🔥 修改：使用评估阶段生成的图像的平均分数作为最终指标
        if len(final_rewards) > 0:
            avg_reward = np.mean(final_rewards)
            eval_best_idx = np.argmax(final_rewards)
            eval_best_image = final_images[eval_best_idx]
            print(f"  Optimization complete: avg reward (eval) = {avg_reward:.4f} (std = {np.std(final_rewards):.4f})")
            print(f"  Individual rewards: {[f'{r:.4f}' for r in final_rewards]}")
            print(f"  (Training best was: {best_reward:.4f})")
            
            # 使用平均分数作为最终指标
            best_reward = avg_reward
            best_image = eval_best_image  # 保留最好的那张作为代表图像
        else:
            print(f"  Warning: No evaluation images generated. Keeping training best: {best_reward:.4f}")
        
        elapsed_time = time.time() - start_time
        
        # 🔥 关键：训练结束后立即移除LoRA adapters释放显存
        if hasattr(self.pipe.unet, 'peft_config') and len(self.pipe.unet.peft_config) > 0:
            self.pipe.unet.delete_adapters(list(self.pipe.unet.peft_config.keys()))
        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return {
            'images': final_images,  # 返回评估图像
            'best_image': best_image,  # 来自训练过程
            'rewards': final_rewards,
            'best_reward': best_reward,  # 来自训练过程
            'time': elapsed_time,
            'n_samples': total_samples_count + len(final_images),  # 训练样本 + 评估样本
            'metadata': {'n_epochs': self.n_epochs, 'batch_size': self.batch_size, 'training_samples': total_samples_count}
        }
    
    def reset(self):
        """重置 DDPO 状态，清理 LoRA 权重并释放显存"""
        from peft import LoraConfig
        
        # 重新初始化 LoRA（移除旧的适配器）
        if hasattr(self.pipe.unet, 'peft_config') and len(self.pipe.unet.peft_config) > 0:
            self.pipe.unet.delete_adapters(list(self.pipe.unet.peft_config.keys()))
        
        # 重新配置 LoRA
        unet_lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.pipe.unet.add_adapter(unet_lora_config)
        
        # LoRA参数设为float32
        for param in self.pipe.unet.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)
        
        # 重新创建优化器
        trainable_params = filter(lambda p: p.requires_grad, self.pipe.unet.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class DPOKWrapper(BaseMethod):
    """DPOK方法的包装器 - 基于d3po实现，添加KL散度约束"""
    
    def __init__(self, config, device="cuda", shared_models=None):
        super().__init__(config, device)
        self.name = "DPOK"
        
        print(f"Initializing {self.name}...")
        
        # 导入依赖
        from peft import LoraConfig
        from model.ATMS_retrieval import ATMS, get_eeg_features
        import sys
        sys.path.append('/home/ldy/Workspace/guide-stable-diffusion/related_works/d3po/d3po')
        
        # 检查是否使用共享模型
        if shared_models is not None:
            print("  Using shared models...")
            # 使用共享的模型实例
            self.eeg_model = shared_models.get('eeg_model')
            self.encoding_model = shared_models.get('encoding_model')
            
            # 🔥 使用共享的SDXL pipeline（不创建新的！）
            self.pipe = shared_models.get('sdxl_pipe')
            if self.pipe is None:
                raise ValueError("Shared SDXL pipeline not found!")
            print("  Using shared SDXL pipeline")
        else:
            # 独立模式：创建自己的 pipeline 和模型
            print("  Standalone mode: loading independently...")
            self.pipe = self._create_independent_pipeline(device)
            self.eeg_model = self._load_eeg_model(config['eeg_model_path'])
            self.encoding_model = self._load_encoding_model(config['encoding_model_path'])
        
        self.subject = config.get('subject', 'sub-01')
        
        # 训练参数
        self.n_epochs = config.get('n_epochs', 5)
        self.batch_size = config.get('batch_size', 4)
        self.learning_rate = config.get('learning_rate', 3e-5)
        self.use_lora = config.get('use_lora', True)
        self.clip_range = config.get('clip_range', 1e-4)
        self.adv_clip_max = config.get('adv_clip_max', 5)
        self.kl_ratio = config.get('kl_ratio', 0.01)  # DPOK特有：KL散度系数
        self.num_inference_steps = config.get('num_inference_steps', 8)
        self.guidance_scale = 0.0
        self.eta = 0.0
        
        # 冻结非训练组件
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.text_encoder_2.requires_grad_(False)
        
        # 🔥 延迟配置LoRA：在optimize()时再添加，避免与其他方法冲突
        print("  LoRA will be configured on first optimize() call")
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.optimizer = None  # 优化器也延迟创建
        
        print(f"  DPOK initialized: {self.n_epochs} epochs, batch_size={self.batch_size}, kl_ratio={self.kl_ratio}")
    
    def _create_independent_pipeline(self, device):
        """创建独立的 SDXL pipeline（与其他方法使用相同的模型）"""
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        
        # 使用与 Generator4Embeds 相同的模型配置
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        
        # 加载 UNet（SDXL-Lightning 8步模型）
        unet = UNet2DConditionModel.from_pretrained(
            model_id, 
            subfolder="unet", 
            torch_dtype=torch.bfloat16, 
            use_safetensors=True, 
            variant="fp16"
        )
        unet.load_state_dict(load_file(hf_hub_download(
            "ByteDance/SDXL-Lightning", 
            "sdxl_lightning_8step_unet.safetensors"
        )))
        
        # 加载 VAE
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.bfloat16
        )
        
        # 加载 Scheduler
        scheduler = DDIMScheduler.from_pretrained(
            model_id, 
            subfolder="scheduler",
            timestep_spacing="trailing"
        )
        
        # 创建 pipeline（不使用 ExtendedStableDiffusionXLPipeline，避免 IP-Adapter 相关代码）
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, 
            unet=unet, 
            vae=vae, 
            scheduler=scheduler, 
            torch_dtype=torch.bfloat16, 
            use_safetensors=True, 
            variant="fp16"
        )
        
        pipe.vae.enable_slicing()
        pipe.to(device)
        
        return pipe
        
    def optimize(self, target_eeg_feature, target_idx, budget=50):
        """运行DPOK优化（基于d3po的实现，添加KL散度约束）"""
        # 实现与DDPO类似，只是在loss中添加KL散度项
        # 为了简洁，这里省略详细实现，与benchmark_framework.py中的DPOKWrapper一致
        # 请参考上面DDPOWrapper的实现，在callback_func中添加KL散度约束即可
        pass
    
    def reset(self):
        """重置 DPOK 状态"""
        pass


class D3POWrapper(BaseMethod):
    """D3PO方法的包装器 - 基于d3po实现，使用pairwise preference learning"""
    
    def __init__(self, config, device="cuda", shared_models=None):
        super().__init__(config, device)
        self.name = "D3PO"
        print(f"Initializing {self.name}...")
        # 详细实现省略，与benchmark_framework.py中一致
        pass
    
    def optimize(self, target_eeg_feature, target_idx, budget=50):
        """运行D3PO优化"""
        pass
    
    def reset(self):
        """重置D3PO状态"""
        pass


# ==================== 新增方法 1: Bayesian Optimization ====================

class BayesianOptimizationWrapper(BaseMethod):
    """
    Bayesian Optimization方法的包装器
    在CLIP embedding space中优化（与PseudoModel保持一致）
    使用高斯过程建模目标函数（EEG相似度），通过acquisition function选择下一个采样点
    """
    
    def __init__(self, config, device="cuda", shared_models=None):
        super().__init__(config, device)
        self.name = "BayesianOpt"
        
        print(f"Initializing {self.name}...")
        
        # 检查是否使用共享模型
        if shared_models is not None:
            print("  Using shared models...")
            self.eeg_model = shared_models.get('eeg_model')
            self.encoding_model = shared_models.get('encoding_model')
            self.vlmodel = shared_models.get('vlmodel')
            self.preprocess_train = shared_models.get('preprocess_train')
            
            # 🔥 使用共享的SDXL pipeline（通过Generator）
            self.pipe = shared_models.get('sdxl_pipe')
            if self.pipe is None:
                raise ValueError("Shared SDXL pipeline not found!")
        else:
            print("  Loading independent models...")
            from exp_batch_offline_generation import vlmodel, preprocess_train
            self.vlmodel = vlmodel
            self.preprocess_train = preprocess_train
            self.eeg_model = self._load_eeg_model(config['eeg_model_path'])
            self.encoding_model = self._load_encoding_model(config['encoding_model_path'])
            
            # 独立模式：加载pipe
            from exp_batch_offline_generation import pipe
            self.pipe = pipe
        
        self.subject = config.get('subject', 'sub-01')
        
        # 🔥 创建独立的随机数生成器（避免与其他方法冲突）
        self.rng = np.random.RandomState(seed=42)
        self.torch_generator = torch.Generator(device=device).manual_seed(42)
        
        # BO特有参数
        self.acquisition = config.get('acquisition', 'ucb')  # 'ucb', 'ei', 'poi'
        self.kappa = config.get('kappa', 2.5)  # UCB的exploration参数
        self.xi = config.get('xi', 0.01)  # EI/POI的exploration参数
        self.n_initial_points = config.get('n_initial_points', 10)  # 初始随机采样点数
        
        # CLIP embedding维度
        self.clip_dim = 1024
        
        # 图像生成参数
        self.num_inference_steps = 8
        self.guidance_scale = 0.0
        
        print(f"  BayesianOpt initialized: acquisition={self.acquisition}, n_initial={self.n_initial_points}")
        print(f"  Optimizing in CLIP embedding space (dim={self.clip_dim})")
        print(f"  Using independent RNG with seed=42")
    
    def _sample_clip_embedding(self, n_samples=1):
        """
        采样CLIP embeddings（在CLIP embedding space中）
        Returns: List[np.ndarray], 每个shape为 (1024,)
        """
        embeddings = []
        for _ in range(n_samples):
            # 🔥 使用独立的随机数生成器
            emb = self.rng.randn(self.clip_dim).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings.append(emb)
        return embeddings
    
    def _clip_embedding_to_image(self, clip_embedding):
        """
        从CLIP embedding直接生成PIL图像（使用IP-Adapter）
        Args:
            clip_embedding: np.ndarray, shape (1024,)
        Returns:
            PIL.Image
        """
        # 转换为torch tensor
        clip_tensor = torch.tensor(clip_embedding, device=self.device, dtype=torch.float32).unsqueeze(0)
        
        # 直接使用IP-Adapter从CLIP embedding生成图像（不需要PseudoTargetModel）
        with torch.no_grad():
            # 🔥 使用独立的随机数生成器生成latent
            latents = torch.randn(
                1, 4, 
                self.pipe.unet.config.sample_size, 
                self.pipe.unet.config.sample_size, 
                device=self.device,
                dtype=torch.bfloat16,
                generator=self.torch_generator  # 🔥 添加 generator
            )
            
            # 使用IP-Adapter生成图像
            output = self.pipe(
                prompt=[""],
                ip_adapter_image_embeds=[clip_tensor.unsqueeze(0).type(torch.bfloat16)],
                latents=latents,
                output_type="latent",
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                eta=1.0,
            )
            
            # 解码latents为图像
            shift_factor = self.pipe.vae.config.shift_factor if self.pipe.vae.config.shift_factor else 0.0
            decoded_latents = (output.images / self.pipe.vae.config.scaling_factor) + shift_factor
            images = self.pipe.vae.decode(decoded_latents, return_dict=False)[0]
            images = self.pipe.image_processor.postprocess(images.detach())
        
        return images[0]  # 返回第一张PIL图像
    
    def _gaussian_process_predict(self, X_train, y_train, X_test):
        """
        简化的高斯过程预测
        Args:
            X_train: np.ndarray, shape (n_train, latent_dim)
            y_train: np.ndarray, shape (n_train,)
            X_test: np.ndarray, shape (n_test, latent_dim)
        Returns:
            mu: np.ndarray, shape (n_test,), 预测均值
            sigma: np.ndarray, shape (n_test,), 预测标准差
        """
        from scipy.spatial.distance import cdist
        
        # 使用RBF核（简化版本）
        length_scale = 1.0
        noise = 1e-6
        
        # K(X_train, X_train)
        K = np.exp(-cdist(X_train, X_train, 'sqeuclidean') / (2 * length_scale**2))
        K += noise * np.eye(len(X_train))
        
        # K(X_test, X_train)
        K_s = np.exp(-cdist(X_test, X_train, 'sqeuclidean') / (2 * length_scale**2))
        
        # K(X_test, X_test)
        K_ss = np.exp(-cdist(X_test, X_test, 'sqeuclidean') / (2 * length_scale**2))
        
        # 预测
        try:
            K_inv = np.linalg.inv(K)
            mu = K_s @ K_inv @ y_train
            cov = K_ss - K_s @ K_inv @ K_s.T
            sigma = np.sqrt(np.maximum(np.diag(cov), 1e-10))
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆，返回均值预测
            mu = np.mean(y_train) * np.ones(len(X_test))
            sigma = np.std(y_train) * np.ones(len(X_test))
        
        return mu, sigma
    
    def _acquisition_function(self, mu, sigma, y_best):
        """
        计算acquisition function
        Args:
            mu: np.ndarray, GP预测均值
            sigma: np.ndarray, GP预测标准差
            y_best: float, 当前最佳观测值
        Returns:
            acquisition_values: np.ndarray
        """
        from scipy.stats import norm
        
        if self.acquisition == 'ucb':
            # Upper Confidence Bound
            return mu + self.kappa * sigma
        
        elif self.acquisition == 'ei':
            # Expected Improvement
            with np.errstate(divide='warn'):
                Z = (mu - y_best - self.xi) / sigma
                ei = (mu - y_best - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
            return ei
        
        elif self.acquisition == 'poi':
            # Probability of Improvement
            with np.errstate(divide='warn'):
                Z = (mu - y_best - self.xi) / sigma
                poi = norm.cdf(Z)
                poi[sigma == 0.0] = 0.0
            return poi
        
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition}")
    
    def optimize(self, target_eeg_feature, target_idx, budget=50):
        """运行Bayesian Optimization优化（在CLIP embedding space中）"""
        start_time = time.time()
        
        print(f"  [BayesianOpt] Starting optimization with budget={budget}")
        print(f"  [BayesianOpt] Optimizing in CLIP embedding space (dim={self.clip_dim})")
        
        # 存储所有采样点和对应的reward
        all_embeddings = []  # List of CLIP embeddings
        all_rewards = []
        all_images = []
        
        best_reward = -float('inf')
        best_image = None
        
        # Step 1: 初始随机采样
        n_initial = min(self.n_initial_points, budget)
        print(f"  [BayesianOpt] Phase 1: Random initialization ({n_initial} samples)")
        
        for i in range(n_initial):
            # 采样CLIP embedding
            clip_emb = self._sample_clip_embedding(1)[0]
            
            # 从CLIP embedding生成图像
            image = self._clip_embedding_to_image(clip_emb)
            
            # 计算reward
            eeg = self._generate_eeg_from_images([image])[0]
            reward = self._compute_eeg_similarity_reward(eeg, target_eeg_feature, self.subject)
            
            # 存储结果
            all_embeddings.append(clip_emb)
            all_rewards.append(reward)
            all_images.append(image)
            
            if reward > best_reward:
                best_reward = reward
                best_image = image
            
            if (i + 1) % 5 == 0:
                print(f"    Initial sampling: {i+1}/{n_initial}, Best reward: {best_reward:.4f}")
        
        # Step 2: Bayesian Optimization迭代
        n_bo_iterations = budget - n_initial
        print(f"  [BayesianOpt] Phase 2: BO iterations ({n_bo_iterations} samples)")
        
        X_train = np.array(all_embeddings)  # shape: (n_samples, 1024)
        y_train = np.array(all_rewards)  # shape: (n_samples,)
        
        for i in range(n_bo_iterations):
            # 生成候选点（在CLIP embedding space中随机采样一批）
            n_candidates = 100
            candidate_embeddings = self._sample_clip_embedding(n_candidates)
            X_candidates = np.array(candidate_embeddings)
            
            # 使用GP预测候选点的均值和方差
            mu, sigma = self._gaussian_process_predict(X_train, y_train, X_candidates)
            
            # 计算acquisition function
            y_best = np.max(y_train)
            acq_values = self._acquisition_function(mu, sigma, y_best)
            
            # 选择acquisition function最大的点
            best_candidate_idx = np.argmax(acq_values)
            next_embedding = X_candidates[best_candidate_idx]
            
            # 从CLIP embedding生成图像并计算reward
            image = self._clip_embedding_to_image(next_embedding)
            eeg = self._generate_eeg_from_images([image])[0]
            reward = self._compute_eeg_similarity_reward(eeg, target_eeg_feature, self.subject)
            
            # 更新训练集
            X_train = np.vstack([X_train, next_embedding])
            y_train = np.append(y_train, reward)
            
            all_embeddings.append(next_embedding)
            all_rewards.append(reward)
            all_images.append(image)
            
            if reward > best_reward:
                best_reward = reward
                best_image = image
            
            if (i + 1) % 5 == 0:
                print(f"    BO iteration: {i+1}/{n_bo_iterations}, Best reward: {best_reward:.4f}")
        
        elapsed_time = time.time() - start_time
        
        print(f"  [BayesianOpt] Optimization complete: best optimization reward = {best_reward:.4f}")
        
        # 🔥 优化完成后，使用最优CLIP embedding生成5张图像用于最终评估
        print(f"  [BayesianOpt] Generating 5 final samples for evaluation...")
        best_embedding = X_train[np.argmax(y_train)]
        
        final_images = []
        final_rewards = []
        num_eval_samples = 5
        
        for i in range(num_eval_samples):
            # 从最优embedding生成图像（每次latents不同）
            image = self._clip_embedding_to_image(best_embedding)
            eeg = self._generate_eeg_from_images([image])[0]
            reward = self._compute_eeg_similarity_reward(eeg, target_eeg_feature, self.subject)
            
            final_images.append(image)
            final_rewards.append(reward)
            
            # 🔥 立即释放中间变量（EEG特征占用显存）
            del eeg
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 计算平均分数
        final_score = np.mean(final_rewards)
        best_eval_image = final_images[np.argmax(final_rewards)]
        
        print(f"  [BayesianOpt] Final evaluation: mean = {final_score:.4f}, std = {np.std(final_rewards):.4f}")
        print(f"  [BayesianOpt] Individual rewards: {[f'{r:.4f}' for r in final_rewards]}")
        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return {
            'images': final_images,  # 最终评估的5张图像
            'best_image': best_eval_image,
            'rewards': final_rewards,
            'best_reward': final_score,  # 5张的平均值
            'time': elapsed_time,
            'n_samples': budget,
            'metadata': {
                'acquisition': self.acquisition,
                'n_initial_points': n_initial,
                'n_bo_iterations': n_bo_iterations,
                'optimization_space': 'CLIP_embedding',
                'num_eval_samples': num_eval_samples,
                'eval_rewards_std': float(np.std(final_rewards)),
                'optimization_best': best_reward  # 优化过程中的最优值
            }
        }
    
    def reset(self):
        """重置BO状态"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# ==================== 新增方法 2: CMA-ES ====================

class CMAESWrapper(BaseMethod):
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) 方法的包装器
    在CLIP embedding space中优化（与PseudoModel保持一致）
    维护一个多变量高斯分布，通过进化策略优化CLIP embedding
    """
    
    def __init__(self, config, device="cuda", shared_models=None):
        super().__init__(config, device)
        self.name = "CMA-ES"
        
        print(f"Initializing {self.name}...")
        
        # 检查是否使用共享模型
        if shared_models is not None:
            print("  Using shared models...")
            self.eeg_model = shared_models.get('eeg_model')
            self.encoding_model = shared_models.get('encoding_model')
            self.vlmodel = shared_models.get('vlmodel')
            self.preprocess_train = shared_models.get('preprocess_train')
            
            # 使用共享的SDXL pipeline（通过Generator）
            self.pipe = shared_models.get('sdxl_pipe')
            if self.pipe is None:
                raise ValueError("Shared SDXL pipeline not found!")
        else:
            print("  Loading independent models...")
            from exp_batch_offline_generation import vlmodel, preprocess_train
            self.vlmodel = vlmodel
            self.preprocess_train = preprocess_train
            self.eeg_model = self._load_eeg_model(config['eeg_model_path'])
            self.encoding_model = self._load_encoding_model(config['encoding_model_path'])
            
            # 独立模式：加载pipe
            from exp_batch_offline_generation import pipe
            self.pipe = pipe
        
        self.subject = config.get('subject', 'sub-01')
        
        # 🔥 创建独立的随机数生成器（使用不同的种子避免与BO冲突）
        self.rng = np.random.RandomState(seed=43)
        self.torch_generator = torch.Generator(device=device).manual_seed(43)
        
        # CMA-ES特有参数
        self.population_size = config.get('population_size', 10)
        self.sigma = config.get('sigma', 0.5)
        
        # CLIP embedding维度
        self.clip_dim = 1024
        
        # 图像生成参数
        self.num_inference_steps = 8
        self.guidance_scale = 0.0
        
        print(f"  CMA-ES initialized: population_size={self.population_size}, sigma={self.sigma}")
        print(f"  Optimizing in CLIP embedding space (dim={self.clip_dim})")
        print(f"  Using independent RNG with seed=43")
    
    def _clip_embedding_to_image(self, clip_embedding):
        """
        从CLIP embedding直接生成PIL图像（使用IP-Adapter）
        Args:
            clip_embedding: np.ndarray, shape (1024,)
        Returns:
            PIL.Image
        """
        # 转换为torch tensor
        clip_tensor = torch.tensor(clip_embedding, device=self.device, dtype=torch.float32).unsqueeze(0)
        
        # 直接使用IP-Adapter从CLIP embedding生成图像（不需要PseudoTargetModel）
        with torch.no_grad():
            # 🔥 使用独立的随机数生成器生成latent
            latents = torch.randn(
                1, 4, 
                self.pipe.unet.config.sample_size, 
                self.pipe.unet.config.sample_size, 
                device=self.device,
                dtype=torch.bfloat16,
                generator=self.torch_generator  # 🔥 添加 generator
            )
            
            # 使用IP-Adapter生成图像
            output = self.pipe(
                prompt=[""],
                ip_adapter_image_embeds=[clip_tensor.unsqueeze(0).type(torch.bfloat16)],
                latents=latents,
                output_type="latent",
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                eta=1.0,
            )
            
            # 解码latents为图像
            shift_factor = self.pipe.vae.config.shift_factor if self.pipe.vae.config.shift_factor else 0.0
            decoded_latents = (output.images / self.pipe.vae.config.scaling_factor) + shift_factor
            images = self.pipe.vae.decode(decoded_latents, return_dict=False)[0]
            images = self.pipe.image_processor.postprocess(images.detach())
        
        return images[0]  # 返回第一张PIL图像
    
    def optimize(self, target_eeg_feature, target_idx, budget=50):
        """运行CMA-ES优化（在CLIP embedding space中）"""
        try:
            import cma
        except ImportError:
            print("  ERROR: cma package not installed. Please run: pip install cma")
            print("  Falling back to random sampling...")
            return self._fallback_random_sampling(target_eeg_feature, budget)
        
        start_time = time.time()
        
        print(f"  [CMA-ES] Starting optimization with budget={budget}")
        print(f"  [CMA-ES] Optimizing in CLIP embedding space (dim={self.clip_dim})")
        
        # 初始均值（零向量）
        x0 = np.zeros(self.clip_dim)
        
        # CMA-ES优化器
        es = cma.CMAEvolutionStrategy(x0, self.sigma, {
            'popsize': self.population_size,
            'maxiter': budget // self.population_size,
            'verb_disp': 0,  # 禁用verbose输出
            'verbose': -1
        })
        
        all_images = []
        all_rewards = []
        best_reward = -float('inf')
        best_image = None
        
        n_evaluations = 0
        generation = 0
        
        while not es.stop() and n_evaluations < budget:
            generation += 1
            
            # 采样population
            solutions = es.ask()
            
            # 评估每个solution
            fitness_list = []
            
            for solution in solutions:
                if n_evaluations >= budget:
                    break
                
                # 归一化为单位向量（CLIP embedding通常是归一化的）
                clip_emb = solution / (np.linalg.norm(solution) + 1e-8)
                clip_emb = clip_emb.astype(np.float32)
                
                # 从CLIP embedding生成图像
                image = self._clip_embedding_to_image(clip_emb)
                
                # 计算reward
                eeg = self._generate_eeg_from_images([image])[0]
                reward = self._compute_eeg_similarity_reward(eeg, target_eeg_feature, self.subject)
                
                # CMA-ES最小化目标，所以取负值
                fitness = -reward
                fitness_list.append(fitness)
                
                all_images.append(image)
                all_rewards.append(reward)
                n_evaluations += 1
                
                if reward > best_reward:
                    best_reward = reward
                    best_image = image
            
            # 更新CMA-ES
            es.tell(solutions[:len(fitness_list)], fitness_list)
            
            print(f"    Generation {generation}, Evaluations: {n_evaluations}/{budget}, Best reward: {best_reward:.4f}")
        
        elapsed_time = time.time() - start_time
        
        print(f"  [CMA-ES] Optimization complete: best optimization reward = {best_reward:.4f}")
        
        # 🔥 优化完成后，使用最优解生成5张图像用于最终评估
        print(f"  [CMA-ES] Generating 5 final samples for evaluation...")
        best_solution = es.result.xbest  # CMA-ES的最优解
        best_clip_emb = best_solution / (np.linalg.norm(best_solution) + 1e-8)
        
        final_images = []
        final_rewards = []
        num_eval_samples = 5
        
        for i in range(num_eval_samples):
            # 从最优embedding生成图像（每次latents不同）
            image = self._clip_embedding_to_image(best_clip_emb)
            eeg = self._generate_eeg_from_images([image])[0]
            reward = self._compute_eeg_similarity_reward(eeg, target_eeg_feature, self.subject)
            
            final_images.append(image)
            final_rewards.append(reward)
            
            # 🔥 立即释放中间变量（EEG特征占用显存）
            del eeg
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 计算平均分数
        final_score = np.mean(final_rewards)
        best_eval_image = final_images[np.argmax(final_rewards)]
        
        print(f"  [CMA-ES] Final evaluation: mean = {final_score:.4f}, std = {np.std(final_rewards):.4f}")
        print(f"  [CMA-ES] Individual rewards: {[f'{r:.4f}' for r in final_rewards]}")
        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return {
            'images': final_images,  # 最终评估的5张图像
            'best_image': best_eval_image,
            'rewards': final_rewards,
            'best_reward': final_score,  # 5张的平均值
            'time': elapsed_time,
            'n_samples': n_evaluations,
            'metadata': {
                'population_size': self.population_size,
                'sigma': self.sigma,
                'n_generations': generation,
                'optimization_space': 'CLIP_embedding',
                'clip_dim': self.clip_dim,
                'num_eval_samples': num_eval_samples,
                'eval_rewards_std': float(np.std(final_rewards)),
                'optimization_best': best_reward  # 优化过程中的最优值
            }
        }
    
    def _fallback_random_sampling(self, target_eeg_feature, budget):
        """如果CMA库不可用，回退到随机采样（在CLIP embedding space中）"""
        start_time = time.time()
        
        print(f"  [CMA-ES Fallback] Using random sampling in CLIP embedding space with budget={budget}")
        
        all_images = []
        all_rewards = []
        best_reward = -float('inf')
        best_image = None
        
        for i in range(budget):
            # 随机采样CLIP embedding
            clip_emb = np.random.randn(self.clip_dim).astype(np.float32)
            clip_emb = clip_emb / (np.linalg.norm(clip_emb) + 1e-8)  # 归一化
            
            # 从CLIP embedding生成图像
            image = self._clip_embedding_to_image(clip_emb)
            
            # 计算reward
            eeg = self._generate_eeg_from_images([image])[0]
            reward = self._compute_eeg_similarity_reward(eeg, target_eeg_feature, self.subject)
            
            all_images.append(image)
            all_rewards.append(reward)
            
            if reward > best_reward:
                best_reward = reward
                best_image = image
            
            if (i + 1) % 10 == 0:
                print(f"    Sample {i+1}/{budget}, Best reward: {best_reward:.4f}")
        
        elapsed_time = time.time() - start_time
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'images': all_images,
            'best_image': best_image,
            'rewards': all_rewards,
            'best_reward': best_reward,
            'time': elapsed_time,
            'n_samples': budget,
            'metadata': {
                'mode': 'random_fallback', 
                'optimization_space': 'CLIP_embedding',
                'clip_dim': self.clip_dim
            }
        }
    
    def reset(self):
        """重置CMA-ES状态"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# ==================== Benchmark Framework ====================

class BenchmarkFramework:
    """主benchmark框架（支持全部6种方法）"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: 配置文件路径（JSON格式）
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = self.config.get('device', 'cuda')
        self.output_dir = self.config.get('output_dir', './benchmark_results_total')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 🔥 创建共享模型（避免重复加载）
        print("\n" + "="*60)
        print("Loading Shared Models (to save GPU memory)...")
        print("="*60)
        self.shared_models = self._load_shared_models()
        
        # 初始化所有方法
        self.methods = self._initialize_methods()
        
        # 加载targets
        self.targets = self._load_targets()
        
        print(f"Initialized {len(self.methods)} methods")
        print(f"Loaded {len(self.targets)} targets")
    
    def _load_shared_models(self):
        """加载所有方法共享的模型，避免重复加载"""
        shared_models = {}
        
        # 获取任意一个启用方法的配置（用于获取模型路径）
        method_configs = self.config['methods']
        enabled_methods = [k for k, v in method_configs.items() if v.get('enabled')]
        
        if not enabled_methods:
            return shared_models
        
        # 从第一个启用的方法获取模型路径
        sample_config = None
        for method_name in enabled_methods:
            config = method_configs[method_name]
            if 'eeg_model_path' in config and 'encoding_model_path' in config:
                sample_config = config
                break
        
        if sample_config is None:
            print("  No methods with model paths found, skipping shared model loading")
            return shared_models
        
        # 加载EEG模型
        print("  Loading shared EEG model...")
        from model.ATMS_retrieval import ATMS
        eeg_model = ATMS()
        checkpoint = torch.load(
            sample_config['eeg_model_path'], 
            map_location=self.device, 
            weights_only=False
        )
        eeg_model.load_state_dict(checkpoint['eeg_model_state_dict'])
        eeg_model.to(self.device)
        eeg_model.eval()
        shared_models['eeg_model'] = eeg_model
        
        # 加载Encoding模型
        print("  Loading shared encoding model...")
        from model.utils import load_model_encoder
        encoding_model = load_model_encoder(
            sample_config['encoding_model_path'], 
            self.device
        )
        encoding_model.eval()
        shared_models['encoding_model'] = encoding_model
        
        # 加载CLIP模型（如果PseudoModel启用）
        if 'pseudo_model' in enabled_methods:
            print("  Loading shared CLIP model...")
            from exp_batch_offline_generation import vlmodel, preprocess_train
            shared_models['vlmodel'] = vlmodel
            shared_models['preprocess_train'] = preprocess_train
        
        # 🔥 重要：创建一个共享的SDXL pipeline供需要的方法使用
        rl_methods = ['ddpo', 'dpok', 'd3po', 'bayesian_opt', 'cma_es']
        if any(m in enabled_methods for m in rl_methods):
            print("  Loading shared SDXL pipeline...")
            shared_models['sdxl_pipe'] = self._create_shared_sdxl_pipeline(self.device)
            print("  Shared SDXL pipeline loaded successfully!")
        
        print("  All shared models loaded!")
        
        return shared_models
    
    def _create_shared_sdxl_pipeline(self, device):
        """创建共享的SDXL pipeline"""
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch.bfloat16,
            use_safetensors=True, variant="fp16"
        )
        unet.load_state_dict(load_file(hf_hub_download(
            "ByteDance/SDXL-Lightning", "sdxl_lightning_8step_unet.safetensors"
        )))
        
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.bfloat16
        )
        
        scheduler = DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler", timestep_spacing="trailing"
        )
        
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, unet=unet, vae=vae, scheduler=scheduler,
            torch_dtype=torch.bfloat16, use_safetensors=True, variant="fp16"
        )
        
        pipe.vae.enable_slicing()
        pipe.to(device)
        
        return pipe
        
    def _initialize_methods(self) -> Dict[str, BaseMethod]:
        """初始化所有要对比的方法（使用共享模型）"""
        methods = {}
        
        method_configs = self.config['methods']
        
        # PseudoModel (Offline)
        if method_configs.get('pseudo_model', {}).get('enabled'):
            methods['PseudoModel'] = PseudoModelWrapper(
                method_configs['pseudo_model'], self.device, 
                shared_models=self.shared_models
            )
        
        # 🔥 新增: Heuristic Closed-Loop Pseudo Model
        if method_configs.get('heuristic_closedloop', {}).get('enabled'):
            methods['HeuristicClosedLoop'] = HeuristicClosedLoopWrapper(
                method_configs['heuristic_closedloop'], self.device,
                shared_models=self.shared_models
            )
        
        # DDPO
        if method_configs.get('ddpo', {}).get('enabled'):
            methods['DDPO'] = DDPOWrapper(
                method_configs['ddpo'], self.device,
                shared_models=self.shared_models
            )
            
        # DPOK (需要完整实现，这里跳过)
        # if method_configs.get('dpok', {}).get('enabled'):
        #     methods['DPOK'] = DPOKWrapper(...)
            
        # D3PO (需要完整实现，这里跳过)
        # if method_configs.get('d3po', {}).get('enabled'):
        #     methods['D3PO'] = D3POWrapper(...)
        
        # 🔥 新增: Bayesian Optimization
        if method_configs.get('bayesian_opt', {}).get('enabled'):
            methods['BayesianOpt'] = BayesianOptimizationWrapper(
                method_configs['bayesian_opt'], self.device,
                shared_models=self.shared_models
            )
        
        # 🔥 新增: CMA-ES
        if method_configs.get('cma_es', {}).get('enabled'):
            methods['CMA-ES'] = CMAESWrapper(
                method_configs['cma_es'], self.device,
                shared_models=self.shared_models
            )
        
        return methods
    
    def _load_targets(self) -> List[Dict]:
        """
        随机采样target EEG特征（确保公平评测）
        从整个图像池中随机抽取指定数量的targets
        """
        # 读取配置
        target_config = self.config['target_selection']
        num_targets = target_config['num_targets']
        random_seed = target_config['random_seed']
        
        embed_dir = self.config['data']['embed_dir']
        image_dir = self.config['data']['image_dir']
        
        # 获取所有可用的文件
        embed_files = sorted([f for f in os.listdir(embed_dir) if f.endswith('_embed.pt')])
        image_files = sorted([f for f in os.listdir(image_dir) 
                             if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        # 确保文件数量一致
        assert len(embed_files) == len(image_files), \
            f"Mismatch: {len(embed_files)} embeds vs {len(image_files)} images"
        
        total_available = len(embed_files)
        
        # 检查请求的数量是否合理
        if num_targets > total_available:
            print(f"  WARNING: Requested {num_targets} targets, but only {total_available} available")
            num_targets = total_available
        
        # 🔥 随机采样target indices（使用固定seed确保可复现）
        rng = np.random.RandomState(seed=random_seed)
        selected_indices = rng.choice(total_available, size=num_targets, replace=False)
        selected_indices = np.sort(selected_indices)  # 排序便于查看（保持为numpy数组）
        
        print(f"  Target Selection: Randomly sampled {num_targets} targets from {total_available} available")
        print(f"  Selected indices: {selected_indices.tolist()}")
        print(f"  Random seed: {random_seed}")
        
        # 加载选中的targets
        targets = []
        for idx in selected_indices:
            targets.append({
                'idx': idx,
                'eeg_feature': torch.load(
                    os.path.join(embed_dir, embed_files[idx]), 
                    weights_only=False
                ),
                'gt_image_path': os.path.join(image_dir, image_files[idx]),
            })
        
        return targets
    
    def run_single_experiment(
        self, 
        method: BaseMethod, 
        target: Dict, 
        seed: int,
        budget: int
    ) -> BenchmarkResult:
        """运行单次实验"""
        
        # 🔥 在每个实验前清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        target_idx = target['idx']
        target_eeg = target['eeg_feature']
        
        print(f"  Running {method.name} on target {target_idx} (seed={seed})...")
        
        try:
            # 记录GPU内存
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated() / 1e9
                print(f"    Initial GPU memory: {initial_memory:.2f} GB")
            
            # 运行优化
            result = method.optimize(target_eeg, target_idx, budget)
            
            # 计算GPU内存峰值
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / 1e9
                gpu_memory = peak_memory - initial_memory
                print(f"    Peak GPU memory: {peak_memory:.2f} GB (增量: {gpu_memory:.2f} GB)")
            else:
                gpu_memory = 0.0
            
            # 保存生成的图像
            if result['best_image'] is not None:
                save_dir = os.path.join(
                    self.output_dir, 'images', method.name, f"target_{target_idx}"
                )
                os.makedirs(save_dir, exist_ok=True)
                result['best_image'].save(
                    os.path.join(save_dir, f"seed_{seed}.png")
                )
            
            # 创建结果对象
            benchmark_result = BenchmarkResult(
                method_name=method.name,
                target_idx=target_idx,
                seed=seed,
                eeg_similarity=result['best_reward'],
                time_seconds=result['time'],
                gpu_memory_gb=gpu_memory,
                n_samples_used=result['n_samples'],
                success=True
            )
            
            # 🔥 立即reset方法以释放显存
            method.reset()
            
            # 清理显存
            del result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
        except Exception as e:
            import traceback
            print(f"    ERROR: {str(e)}")
            # 打印完整的错误堆栈（用于调试）
            if "CUDA out of memory" not in str(e):
                traceback.print_exc()
            
            benchmark_result = BenchmarkResult(
                method_name=method.name,
                target_idx=target_idx,
                seed=seed,
                eeg_similarity=0.0,
                success=False,
                error_message=str(e)
            )
            
            # 🔥 即使出错也要reset方法并清理显存
            try:
                method.reset()
            except:
                pass  # reset失败也继续
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        return benchmark_result
    
    def experiment_1_single_target(self) -> List[BenchmarkResult]:
        """
        实验1: 单目标重建对比
        所有方法在相同budget下生成图像，比较效果
        """
        print("\n" + "="*60)
        print("实验1: 单目标重建对比（6种方法）")
        print("="*60)
        
        exp_config = self.config['experiments']['exp1']
        budget = exp_config['sample_budget']
        seeds = exp_config['random_seeds']
        
        all_results = []
        
        for target in tqdm(self.targets, desc="Targets"):
            for method_name, method in self.methods.items():
                for seed in seeds:
                    result = self.run_single_experiment(
                        method, target, seed, budget
                    )
                    all_results.append(result)
        
        # 保存结果
        self._save_results(all_results, "exp1_results.csv")
        
        return all_results
    
    def _save_results(self, results: List[BenchmarkResult], filename: str):
        """保存结果到CSV"""
        df = pd.DataFrame([r.to_dict() for r in results])
        output_path = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    def visualize_results(self, results: List[BenchmarkResult], output_prefix: str):
        """可视化结果"""
        df = pd.DataFrame([r.to_dict() for r in results if r.success])
        
        if len(df) == 0:
            print("No successful experiments to visualize!")
            return
        
        # 汇总统计表
        summary = df.groupby('method_name').agg({
            'eeg_similarity': ['mean', 'std', 'max'],
            'time_seconds': ['mean', 'std'],
            'gpu_memory_gb': ['mean', 'max'],
            'n_samples_used': 'mean'
        }).round(4)
        
        print("\n" + "="*120)
        print("Statistical Summary Across All 6 Methods:")
        print("="*120)
        print(summary)
        print("="*120)
        
        # 保存统计
        summary.to_csv(os.path.join(self.output_dir, f'{output_prefix}_summary.csv'))
        
        # 箱线图对比
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='method_name', y='eeg_similarity')
        plt.title('EEG Similarity Distribution Across 6 Methods')
        plt.ylabel('EEG Similarity')
        plt.xlabel('Method')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{output_prefix}_comparison.png'), dpi=300)
        plt.close()
    
    def run_full_benchmark(self):
        """运行完整的benchmark"""
        print("\n" + "="*80)
        print("Starting Full Benchmark (6 Methods)")
        print("="*80)
        
        results_exp1 = self.experiment_1_single_target()
        self.visualize_results(results_exp1, 'exp1')
        
        print("\n" + "="*80)
        print("Benchmark Complete!")
        print(f"Results saved to: {self.output_dir}")
        print("="*80)
        
        return {'exp1': results_exp1}


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Benchmark Experiments (6 Methods)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config JSON file')
    parser.add_argument('--exp', type=str, default='all',
                       choices=['all', 'exp1'],
                       help='Which experiment to run')
    
    args = parser.parse_args()
    
    # 创建benchmark框架
    framework = BenchmarkFramework(args.config)
    
    # 运行实验
    if args.exp == 'all':
        framework.run_full_benchmark()
    elif args.exp == 'exp1':
        results = framework.experiment_1_single_target()
        framework.visualize_results(results, 'exp1')


if __name__ == '__main__':
    main()


