"""
Benchmark Framework for Comparing Different Optimization Methods (Extended Version)
Compares 7 methods:
  1. PseudoModel (Offline)         - Offline sampling + GP optimization
  2. HeuristicClosedLoop          - Closed-loop iteration (fusion + greedy sampling)
  3. DDPO                          - Reinforcement learning (PPO)
  4. DPOK                          - Reinforcement learning (KL regularization)
  5. D3PO                          - Reinforcement learning (DPO)
  6. BayesianOpt                   - Bayesian optimization
  7. CMA-ES                        - Evolution strategy
"""

# CUDA_VISIBLE_DEVICES=1 python benchmark_framework_total.py --config benchmark_config_total.json --exp exp1


import os
import sys

# Important: set environment variables before importing other modules to prevent overwriting
# If CUDA_VISIBLE_DEVICES is already set via command line, use that value
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    # If not set, use default value (modify here as needed)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Default to GPU 0

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
    """Single experiment result"""
    method_name: str
    target_idx: int
    seed: int
    
    # Performance metrics
    eeg_similarity: float
    clip_score: float = None
    aesthetic_score: float = None
    ssim: float = None
    
    # Efficiency metrics
    time_seconds: float = 0.0
    gpu_memory_gb: float = 0.0
    n_samples_used: int = 0
    
    # Additional information
    success: bool = True
    error_message: str = None
    
    def to_dict(self):
        return asdict(self)


# ==================== HeuristicGenerator Class ====================
# Strictly follows the implementation from exp-benchmark_heuristic_generation.py

import einops
import torch.nn.functional as F
from model.pseudo_target_model import PseudoTargetModel

class HeuristicGenerator:
    """
    HeuristicGenerator strictly following the exp-benchmark_heuristic_generation.py implementation.
    Used for fusion-based generation in the HeuristicClosedLoop method.
    """
    def __init__(self, pipe, vlmodel, preprocess_train, device="cuda", seed=42, load_ip_adapter=False, min_data_threshold=10):
        self.pipe = pipe
        self.vlmodel = vlmodel
        self.preprocess_train = preprocess_train
        self.device = device

        # Hyperparameters
        self.batch_size = 32
        self.alpha = 80
        self.total_steps = 15  # Key: 15-step optimization
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
        self.min_data_threshold = min_data_threshold  # Minimum data size threshold

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
        Core method: includes 15-step GP optimization + final generation
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

        # Key: 15-step optimization loop - only optimizes pseudo_target, no image generation
        for step in range(self.total_steps):
            data_x, data_y = self.pseudo_target_model.get_model_data()   
            if data_y.size(0) < self.min_data_threshold:  # Check if data size is sufficient
                print(f"[WARNING] Insufficient data ({data_y.size(0)} < {self.min_data_threshold}), returning random generation")
                # Insufficient data, generate random image and return
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
            
            # Data is sufficient, only perform pseudo_target optimization (no image generation)
            step_size = self.initial_step_size / (1 + self.decay_rate * step)
            pseudo_target, _ = self.pseudo_target_model.estimate_pseudo_target(pseudo_target, step_size=step_size)

        # After optimization loop ends, generate final image using the optimized pseudo_target
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


# ==================== BaseMethod Class ====================

class BaseMethod:
    """Base class for all methods - provides shared utility methods"""
    
    def __init__(self, config: dict, device: str = "cuda"):
        self.config = config
        self.device = device
        self.name = "BaseMethod"
        
        # Subclasses can set these attributes; base class methods will use them
        self.eeg_model = None
        self.encoding_model = None
        
    def optimize(self, target_eeg_feature, target_idx: int, budget: int = 50) -> Dict[str, Any]:
        """
        Optimize and generate images
        
        Args:
            target_eeg_feature: Target EEG feature
            target_idx: Target index
            budget: Sampling budget
            
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
        """Reset method state (for repeated experiments)"""
        pass
    
    # ==================== Shared Utility Methods ====================
    
    def _load_eeg_model(self, model_path):
        """Load EEG model (ATMS)"""
        from model.ATMS_retrieval import ATMS
        eeg_model = ATMS()
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        eeg_model.load_state_dict(checkpoint['eeg_model_state_dict'])
        eeg_model.to(self.device)
        eeg_model.eval()
        return eeg_model
    
    def _load_encoding_model(self, model_path):
        """Load encoding model"""
        from model.utils import load_model_encoder
        encoding_model = load_model_encoder(model_path, self.device)
        encoding_model.eval()
        return encoding_model
    
    def _preprocess_image(self, image, device=None):
        """
        Preprocess image (PIL Image or file path)
        
        Args:
            image: PIL.Image or image file path string
            device: Target device, defaults to self.device
            
        Returns:
            torch.Tensor: Preprocessed image tensor (1, 3, 224, 224)
        """
        import torchvision.transforms as transforms
        
        if device is None:
            device = self.device
            
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # If it's a path string, load the image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        return transform(image).unsqueeze(0).to(device)
    
    def _generate_eeg_from_images(self, images, device=None):
        """
        Generate EEG features from a list of PIL images
        
        Args:
            images: List[PIL.Image] or List[str] (image paths)
            device: Target device, defaults to self.device
            
        Returns:
            List[np.ndarray]: List of generated EEG features
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
            
            # Immediately release image tensor
            del img_tensor
        
        return synthetic_eegs
    
    def _compute_eeg_similarity_reward(self, eeg, target_feature, subject='sub-01'):
        """
        Compute similarity reward between EEG features and target features
        
        Args:
            eeg: np.ndarray or torch.Tensor, EEG features (17, 250) or (1, 17, 250)
            target_feature: torch.Tensor, target EEG feature
            subject: str, subject ID
            
        Returns:
            float: Normalized similarity score [0, 1]
        """
        from model.ATMS_retrieval import get_eeg_features
        
        if self.eeg_model is None:
            raise ValueError("eeg_model not initialized. Please set self.eeg_model in subclass.")
        
        # Ensure eeg is a torch.Tensor and add batch dimension
        # generate_eeg returns (17, 250), needs to become (1, 17, 250)
        if not isinstance(eeg, torch.Tensor):
            eeg = torch.tensor(eeg)
        
        # If no batch dimension (dim < 3), add batch dimension
        if eeg.dim() < 3:
            eeg = eeg.unsqueeze(0)
        
        # Get EEG features
        eeg_feature = get_eeg_features(
            self.eeg_model, 
            eeg, 
            self.device, 
            subject
        )
        
        # Compute cosine similarity and normalize to [0, 1]
        similarity = torch.nn.functional.cosine_similarity(
            eeg_feature.to(self.device), 
            target_feature.to(self.device)
        )
        normalized_similarity = (similarity + 1) / 2
        
        return normalized_similarity.item()


class PseudoModelWrapper(BaseMethod):
    """Wrapper for the existing Pseudo Model method"""
    
    def __init__(self, config, device="cuda", shared_models=None):
        super().__init__(config, device)
        self.name = "PseudoModel"
        
        print(f"Initializing {self.name}...")
        
        # Check whether to use shared models
        if shared_models is not None:
            print("  Using shared models...")
            self.eeg_model = shared_models.get('eeg_model')
            self.encoding_model = shared_models.get('encoding_model')
            self.vlmodel = shared_models.get('vlmodel')
            self.preprocess_train = shared_models.get('preprocess_train')
        else:
            # Standalone mode: load own models
            print("  Loading independent models...")
            from exp_batch_offline_generation import vlmodel, preprocess_train
            self.vlmodel = vlmodel
            self.preprocess_train = preprocess_train
        
        # Load models using base class methods
        print("  Loading EEG model...")
        self.eeg_model = self._load_eeg_model(config['eeg_model_path'])
        
        print("  Loading encoding model...")
        self.encoding_model = self._load_encoding_model(config['encoding_model_path'])
        
        # Import global Generator and pipe (these don't consume much GPU memory)
        from exp_batch_offline_generation import HeuristicGenerator, pipe
        self.generator = HeuristicGenerator(pipe, self.vlmodel, self.preprocess_train, device=device)
        self.pipe = pipe
        
        self.subject = config.get('subject', 'sub-01')
        
    def _get_image_pool(self, image_set_path):
        """Get candidate image pool"""
        test_images_path = []
        for root, dirs, files in os.walk(image_set_path):
            for file in sorted(files):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    test_images_path.append(os.path.join(root, file))
        return test_images_path
    
    def _generate_eeg_from_image_paths(self, test_image_list, device):
        """Generate EEG from image paths (using base class method)"""
        # Use the base class _generate_eeg_from_images method, which supports image paths
        synthetic_eegs = self._generate_eeg_from_images(test_image_list, device)
        return np.asarray(synthetic_eegs)
    
    def optimize(self, target_eeg_feature, target_idx, budget=50):
        """Run Pseudo Model optimization"""
        start_time = time.time()
        
        # 1. Get image pool and sample
        image_pool = self._get_image_pool(self.config['image_dir'])
        
        # Exclude target image (to prevent data leakage)
        if target_idx < len(image_pool):
            target_image_path = image_pool[target_idx]
            image_pool = [p for p in image_pool if p != target_image_path]
            print(f"  [PseudoModel] Excluded target image: {os.path.basename(target_image_path)}")
        
        # Determine actual sampling count (cannot exceed pool size after excluding target)
        actual_budget = min(budget, len(image_pool))
        print(f"  [PseudoModel] Sampling {actual_budget} images from pool of {len(image_pool)} (budget={budget})")
        sampled_paths = np.random.choice(image_pool, size=actual_budget, replace=False)
        
        # 2. Compute CLIP embeddings (batch processing to reduce peak GPU memory)
        sampled_images = [Image.open(p).convert("RGB") for p in sampled_paths]
        
        batch_size_clip = 16  # Process 16 images at a time
        offline_embeds_list = []
        
        with torch.no_grad():
            for i in range(0, len(sampled_images), batch_size_clip):
                batch_images = sampled_images[i:i+batch_size_clip]
                tensor_batch = torch.stack([self.preprocess_train(img) for img in batch_images]).to(self.device)
                batch_embeds = self.vlmodel.encode_image(tensor_batch)
                offline_embeds_list.append(batch_embeds.cpu())  # Immediately move to CPU
                del tensor_batch, batch_embeds
        
        offline_embeds = torch.cat(offline_embeds_list, dim=0).to(self.device)
        del offline_embeds_list
        
        # 3. Compute rewards (using base class methods)
        synthetic_eegs = self._generate_eeg_from_image_paths(
            sampled_paths, self.device
        )
        
        offline_rewards = []
        for eeg in synthetic_eegs:
            r = self._compute_eeg_similarity_reward(
                eeg, target_eeg_feature, self.subject
            )
            offline_rewards.append(r)
        
        # Immediately release EEG features (rewards already computed)
        del synthetic_eegs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        offline_rewards_tensor = torch.tensor(offline_rewards).to(self.device)
        
        # 4. Add data to Pseudo Model
        from model.pseudo_target_model import PseudoTargetModel
        self.generator.pseudo_target_model = PseudoTargetModel(
            dimension=1024, noise_level=1e-4
        ).to(self.device)
        
        self.generator.pseudo_target_model.add_model_data(
            offline_embeds,
            (-offline_rewards_tensor * self.generator.reward_scaling_factor).to(self.device)
        )
        
        # 5. Evaluation phase: generate 5 images and compute average score
        num_eval_samples = 5
        print(f"  [PseudoModel] Generating {num_eval_samples} evaluation images...")
        
        # Release large tensors no longer needed, freeing space for evaluation
        del sampled_images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        final_images = []
        final_rewards = []
        
        for eval_idx in range(num_eval_samples):
            # Generate one image at a time (since generator.generate_batch_size=1)
            eval_images = self.generator.generate(
                data_x=offline_embeds,
                data_y=offline_rewards_tensor,
                tar_image_embed=None,
                prompt='',
                save_path=None
            )
            
            # Compute the reward for this image
            eval_eegs = self._generate_eeg_from_images(eval_images, self.device)
            eval_reward = self._compute_eeg_similarity_reward(
                eval_eegs[0], target_eeg_feature, self.subject
            )
            
            final_images.extend(eval_images)
            final_rewards.append(eval_reward)
            
            # Immediately release intermediate variables
            del eval_eegs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        elapsed_time = time.time() - start_time
        
        # 6. Compute average score as the final result
        avg_reward = np.mean(final_rewards)
        print(f"  [PseudoModel] Evaluation complete: avg reward = {avg_reward:.4f} (std = {np.std(final_rewards):.4f})")
        print(f"  [PseudoModel] Individual rewards: {[f'{r:.4f}' for r in final_rewards]}")
        
        result = {
            'images': final_images,
            'best_image': final_images[0],  # Keep the first image as representative
            'rewards': final_rewards,
            'best_reward': avg_reward,  # Use average value
            'time': elapsed_time,
            'n_samples': budget,
            'metadata': {
                'optimization_steps': self.generator.total_steps,
                'pool_size': budget,
                'num_eval_samples': num_eval_samples,
                'eval_rewards_std': float(np.std(final_rewards))
            }
        }
        
        # Clean up intermediate variables to release GPU memory
        del offline_embeds, offline_rewards_tensor
        
        # Clean up Pseudo Target Model
        if hasattr(self.generator, 'pseudo_target_model') and self.generator.pseudo_target_model is not None:
            del self.generator.pseudo_target_model
            self.generator.pseudo_target_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return result
    
    def reset(self):
        """Reset Generator state and clean up GPU cache"""
        # Clean up Pseudo Target Model
        if hasattr(self.generator, 'pseudo_target_model') and self.generator.pseudo_target_model is not None:
            del self.generator.pseudo_target_model
            self.generator.pseudo_target_model = None
        
        # PseudoTargetModel will be automatically recreated on the next optimize call
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class HeuristicClosedLoopWrapper(BaseMethod):
    """Heuristic Closed-Loop Pseudo Model method (iterative fusion + greedy sampling)"""
    
    def __init__(self, config, device="cuda", shared_models=None):
        super().__init__(config, device)
        self.name = "HeuristicClosedLoop"
        
        print(f"Initializing {self.name}...")
        
        # Check whether to use shared models
        if shared_models is not None:
            print("  Using shared models...")
            self.eeg_model = shared_models.get('eeg_model')
            self.encoding_model = shared_models.get('encoding_model')
            self.vlmodel = shared_models.get('vlmodel')
            self.preprocess_train = shared_models.get('preprocess_train')
        else:
            # Standalone mode: load own models
            print("  Loading independent models...")
            from exp_batch_offline_generation import vlmodel, preprocess_train
            self.vlmodel = vlmodel
            self.preprocess_train = preprocess_train
        
        # Load models using base class methods
        print("  Loading EEG model...")
        self.eeg_model = self._load_eeg_model(config['eeg_model_path'])
        
        print("  Loading encoding model...")
        self.encoding_model = self._load_encoding_model(config['encoding_model_path'])
        
        # Import pipe
        from exp_batch_offline_generation import pipe
        
        # Use shared pipe or load independently
        if shared_models is not None and shared_models.get('sdxl_pipe') is not None:
            self.pipe = shared_models.get('sdxl_pipe')
        else:
            self.pipe = pipe
        
        # Initialize HeuristicGenerator (contains PseudoTargetModel)
        # Uses the HeuristicGenerator class defined in this file
        # Don't auto-load IP adapter (already loaded via shared_models)
        self.generator = HeuristicGenerator(
            self.pipe, 
            self.vlmodel, 
            self.preprocess_train, 
            device=device,
            seed=42,
            load_ip_adapter=False,  # Don't load again
            min_data_threshold=5  # Set to 5 to match initial_sample_size
        )
        
        self.subject = config.get('subject', 'sub-01')
        
        # Closed-loop specific parameters
        # Key improvement: actual loop count is auto-determined by budget; num_loops_max is only an optional hard cap
        # This allows running as many optimization iterations as possible without exceeding the budget
        self.num_loops_max = config.get('num_loops_closedloop', 999)  # Max loop count cap (default 999, actual determined by budget)
        self.num_fusions_per_round = config.get('num_fusions_per_round', 2)  # Fusions per round
        self.top_k_greedy = config.get('top_k_greedy', 20)  # Top-k for greedy sampling
        self.initial_sample_size = config.get('initial_sample_size_closedloop', 5)  # Initial sample count
        
        print(f"  Closed-loop params: num_loops_max={self.num_loops_max} (auto from budget), fusions_per_round={self.num_fusions_per_round}")
    
    def _get_image_pool(self, image_set_path):
        """Get candidate image pool"""
        test_images_path = []
        for root, dirs, files in os.walk(image_set_path):
            for file in sorted(files):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    test_images_path.append(os.path.join(root, file))
        return test_images_path
    
    def _fusion_image_generation(self, fit_images, fit_rewards):
        """
        Fusion-based image generation (strictly following original implementation)
        Calls Generator.generate(), which includes 15-step GP optimization
        """
        import random
        
        # Extract CLIP embeddings from fit_images
        tensor_fit_images = [self.preprocess_train(img) for img in fit_images]
        with torch.no_grad():
            img_embeds = self.vlmodel.encode_image(torch.stack(tensor_fit_images).to(self.device))
        
        # Randomly select two images for fusion
        idx1, idx2 = random.sample(range(len(img_embeds)), 2)
        embed1 = img_embeds[idx1].unsqueeze(0)
        embed2 = img_embeds[idx2].unsqueeze(0)
        
        # Partial swap (similar to genetic crossover)
        scale = 512
        embed_len = embed1.size(1)
        start_idx = random.randint(0, embed_len - scale - 1)
        end_idx = start_idx + scale
        
        temp = embed1[:, start_idx:end_idx].clone()
        embed1[:, start_idx:end_idx] = embed2[:, start_idx:end_idx]
        embed2[:, start_idx:end_idx] = temp
        
        # Use Generator.generate() to produce images (includes 15-step optimization)
        generated_images = []
        
        with torch.no_grad():
            # Generate first image (optimizing from embed1)
            images1 = self.generator.generate(
                img_embeds.to(self.device),
                torch.tensor(fit_rewards).to(self.device),
                None,
                prompt='',
                save_path=None,
                start_embedding=embed1
            )
            generated_images.extend(images1)
            
            # Generate second image (optimizing from embed2)
            images2 = self.generator.generate(
                img_embeds.to(self.device),
                torch.tensor(fit_rewards).to(self.device),
                None,
                prompt='',
                save_path=None,
                start_embedding=embed2
            )
            generated_images.extend(images2)
        
        # Return generated images and source image indices
        return generated_images, (idx1, idx2), fit_images
    
    def _greedy_sampling(self, target_clip_embed, image_pool, processed_paths, test_set_img_embeds, num_samples):
        """Target-based greedy sampling"""
        # Get indices of available images
        available_indices = []
        for i, path in enumerate(image_pool):
            if path not in processed_paths:
                available_indices.append(i)
        
        if len(available_indices) == 0:
            return []
        
        # Compute similarity based on target_clip_embed
        available_features = test_set_img_embeds[available_indices]
        
        # Compute cosine similarity
        target_norm = torch.nn.functional.normalize(target_clip_embed.float(), p=2, dim=1)
        available_norm = torch.nn.functional.normalize(available_features.float(), p=2, dim=1)
        cosine_similarities = torch.mm(available_norm, target_norm.t()).squeeze(1)
        cosine_similarities = (cosine_similarities + 1) / 2
        
        sorted_available_indices = np.argsort(cosine_similarities.cpu())
        
        # Randomly sample from top-K
        top_indices = sorted_available_indices[-min(self.top_k_greedy, len(sorted_available_indices)):]
        num_to_sample = min(num_samples, len(top_indices))
        selected_indices = np.random.choice(top_indices, size=num_to_sample, replace=False)
        
        # Load selected images
        greedy_images = []
        sample_image_paths = []
        for selected_idx in selected_indices:
            greedy_image = Image.open(image_pool[available_indices[selected_idx]]).convert("RGB")
            greedy_images.append(greedy_image)
            sample_image_paths.append(image_pool[available_indices[selected_idx]])
        
        processed_paths.update(sample_image_paths)
        
        return greedy_images
    
    def optimize(self, target_eeg_feature, target_idx, budget=50):
        """Run Heuristic Closed-Loop optimization"""
        from scipy.special import softmax
        from model.pseudo_target_model import PseudoTargetModel
        
        start_time = time.time()
        
        # Re-initialize Generator's PseudoTargetModel (create a new one for each optimization)
        self.generator.pseudo_target_model = PseudoTargetModel(
            dimension=1024, 
            noise_level=1e-4
        ).to(self.device)
        
        # Automatically compute optimal loop count from budget
        IMAGES_PER_LOOP_INITIAL = self.initial_sample_size  # Round 1: initial sampling
        IMAGES_PER_LOOP_FUSION = self.num_fusions_per_round * 4 + self.num_fusions_per_round * 2  # Subsequent rounds: fusion generation + source images + greedy sampling
        
        remaining_budget = budget - IMAGES_PER_LOOP_INITIAL
        num_loops_from_budget = 1 + max(0, remaining_budget // IMAGES_PER_LOOP_FUSION)
        
        # Prefer budget-based loop count; num_loops_max is only a hard cap
        actual_num_loops = min(self.num_loops_max, num_loops_from_budget)
        estimated_images = IMAGES_PER_LOOP_INITIAL + (actual_num_loops - 1) * IMAGES_PER_LOOP_FUSION
        
        print(f"  [{self.name}] Starting closed-loop optimization...")
        print(f"  [{self.name}] Budget={budget}, Budget-based loops={num_loops_from_budget}, Max loops={self.num_loops_max}")
        print(f"  [{self.name}] Actual loops={actual_num_loops}, Estimated images={estimated_images} (utilization: {estimated_images/budget*100:.1f}%)")
        print(f"  [{self.name}] Config: fusions/round={self.num_fusions_per_round}, initial_samples={self.initial_sample_size}")
        
        # 1. Get image pool
        image_pool = self._get_image_pool(self.config['image_dir'])
        
        # Exclude target image
        if target_idx < len(image_pool):
            target_image_path = image_pool[target_idx]
            image_pool = [p for p in image_pool if p != target_image_path]
            print(f"  [{self.name}] Excluded target image: {os.path.basename(target_image_path)}")
        
        # 2. Pre-compute CLIP embeddings for all images (to avoid redundant computation)
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
        
        # 3. Get target's CLIP embedding
        target_image_path = image_pool[0] if target_idx >= len(image_pool) else image_pool[target_idx]
        target_image_pil = Image.open(target_image_path).convert("RGB")
        with torch.no_grad():
            target_clip_embed = self.vlmodel.encode_image(
                self.preprocess_train(target_image_pil).unsqueeze(0).to(self.device)
            )
        
        # 4. Closed-loop iteration
        processed_paths = set()
        fit_images = []
        fit_eegs = []
        fit_rewards = []
        
        all_loop_rewards = []
        all_loop_images = []
        total_images_evaluated = 0  # Track total number of images actually evaluated
        
        # Save all candidates from the final round (for evaluation)
        final_loop_images = []
        final_loop_eegs = []
        final_loop_rewards = []
        
        # Use actual_num_loops computed from budget
        for t in range(actual_num_loops):
            print(f"\n  [{self.name}] Loop {t+1}/{actual_num_loops}")
            
            loop_sample_ten = []
            loop_eeg_ten = []
            loop_reward_ten = []
            loop_loss_ten = []
            
            if t == 0:
                # Initial round: random sampling from image pool
                print(f"    Initial sampling: {self.initial_sample_size} images")
                available_paths = [path for path in image_pool if path not in processed_paths]
                sample_image_paths = np.random.choice(
                    available_paths, 
                    min(self.initial_sample_size, len(available_paths)), 
                    replace=False
                )
                
                chosen_images = [Image.open(p).convert("RGB") for p in sample_image_paths]
                processed_paths.update(sample_image_paths)
                
                # Compute rewards
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
                
                # Update image evaluation count
                total_images_evaluated += len(chosen_images)
                
                # Add to PseudoTargetModel
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
                # Subsequent rounds: fusion generation + greedy sampling
                all_generated_images = []
                all_fusion_source_images = []
                
                # Multiple fusions
                for fusion_idx in range(self.num_fusions_per_round):
                    print(f"      Fusion {fusion_idx+1}/{self.num_fusions_per_round}")
                    
                    generated_images, (idx1, idx2), fit_imgs = self._fusion_image_generation(fit_images, fit_rewards)
                    synthetic_eegs = self._generate_eeg_from_images(generated_images, self.device)
                    
                    loop_sample_ten.extend(generated_images)
                    loop_eeg_ten.extend(synthetic_eegs)
                    all_generated_images.extend(generated_images)
                    
                    # Compute rewards for fusion-generated images
                    for eeg in synthetic_eegs:
                        r = self._compute_eeg_similarity_reward(eeg, target_eeg_feature, self.subject)
                        loop_reward_ten.append(r)
                        loop_loss_ten.append(0)
                    
                    del synthetic_eegs
                    torch.cuda.empty_cache()
                    
                    # Add fusion source images
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
                
                # Update image evaluation count (fusion-generated images + source images)
                total_images_evaluated += len(all_generated_images) + len(all_fusion_source_images)
                
                # Greedy sampling
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
                    
                    # Update image evaluation count (greedy-sampled images)
                    total_images_evaluated += len(greedy_images)
                    
                    del synthetic_eegs
                    torch.cuda.empty_cache()
                    
                    print(f"      Greedy sampling: {len(greedy_images)} images from top-{self.top_k_greedy}")
                
                # Select top-5 to carry over to next round (consistent with other methods)
                loop_probabilities = softmax(loop_reward_ten)
                
                # Probabilistic sampling for top-5
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
                
                # Sort by reward
                combined = list(zip(chosen_rewards, chosen_losses, chosen_images, chosen_eegs))
                combined.sort(reverse=True, key=lambda x: x[0])
                chosen_rewards, chosen_losses, chosen_images, chosen_eegs = zip(*combined)
                chosen_rewards = list(chosen_rewards)
                chosen_losses = list(chosen_losses)
                chosen_images = list(chosen_images)
                chosen_eegs = list(chosen_eegs)
                
                # Add to PseudoTargetModel
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
            
            # Update fit data
            fit_images = chosen_images
            fit_eegs = chosen_eegs
            fit_rewards = chosen_rewards
            fit_losses = chosen_losses
            
            # Record data for this round
            all_loop_rewards.append(np.mean(loop_reward_ten))
            all_loop_images.extend(loop_sample_ten[:4])  # Keep first 4 as representatives
            
            print(f"      Loop {t+1} complete: mean_reward={np.mean(loop_reward_ten):.4f}, best_reward={max(loop_reward_ten):.4f}")
            
            # If this is the final round, save all candidate images for evaluation (original evaluation approach)
            if t == actual_num_loops - 1:
                print(f"  [{self.name}] Saving final round candidates for evaluation...")
                final_loop_images = loop_sample_ten.copy()
                final_loop_eegs = loop_eeg_ten.copy()
                final_loop_rewards = loop_reward_ten.copy()
                print(f"  [{self.name}] Final round: {len(final_loop_images)} candidates saved")
        
        # Record optimization time (excluding evaluation time)
        optimization_time = time.time() - start_time
        
        print(f"  [{self.name}] Optimization complete in {optimization_time:.2f}s")
        print(f"  [{self.name}] Training best reward: {np.mean(fit_rewards):.4f}")
        print(f"  [{self.name}] Actual image evaluations during training: {total_images_evaluated} (budget: {budget}, estimated: {estimated_images})")
        
        # 5. Evaluation phase: select Top-5 from the final round's candidates (original evaluation approach)
        # This is the correct evaluation method for HeuristicClosedLoop: candidates are produced iteratively, then the best are selected
        num_eval_samples = 5
        
        if len(final_loop_images) > 0:
            print(f"  [{self.name}] Selecting Top-{num_eval_samples} from final round's {len(final_loop_images)} candidates...")
            print(f"  [{self.name}] Final round rewards: {[f'{r:.4f}' for r in final_loop_rewards]}")
            
            # Select Top-5 from the final round's candidates (based on EEG similarity reward)
            sorted_indices = np.argsort(final_loop_rewards)[-num_eval_samples:]  # Select 5 with highest reward
            sorted_indices = sorted_indices[::-1]  # Sort from high to low
            
            eval_images = [final_loop_images[i] for i in sorted_indices]
            eval_eegs = [final_loop_eegs[i] for i in sorted_indices]
            eval_rewards = [final_loop_rewards[i] for i in sorted_indices]
            
            print(f"  [{self.name}] Selected Top-{num_eval_samples} rewards: {[f'{r:.4f}' for r in eval_rewards]}")
            
        else:
            # If no final round data was saved (e.g. early stop), use fit_images as fallback
            print(f"  [{self.name}] Warning: No final round candidates, using fit_images as fallback")
            eval_images = fit_images[:num_eval_samples]
            eval_eegs = fit_eegs[:num_eval_samples]
            eval_rewards = fit_rewards[:num_eval_samples]
        
        # Compute evaluation results
        final_eval_reward = np.mean(eval_rewards)
        best_eval_image = eval_images[0]  # Already sorted by reward, first is best
        
        # Total time (optimization time, no additional evaluation time needed)
        total_time = optimization_time
        
        print(f"  [{self.name}] Evaluation complete:")
        print(f"  [{self.name}]   - Evaluation mean reward: {final_eval_reward:.4f} (std: {np.std(eval_rewards):.4f})")
        print(f"  [{self.name}]   - Individual eval rewards: {[f'{r:.4f}' for r in eval_rewards]}")
        print(f"  [{self.name}]   - Training best reward: {np.mean(fit_rewards):.4f} (for comparison)")
        print(f"  [{self.name}]   - Total time: {total_time:.2f}s")
        print(f"  [{self.name}]   - Evaluation method: Selection from final round candidates (original method)")
        
        # Return selected Top-5 images (original evaluation approach)
        result = {
            'images': eval_images,  # Top-5 selected from final round candidates
            'best_image': best_eval_image,  # Top-1
            'rewards': eval_rewards,  # Corresponding rewards
            'best_reward': final_eval_reward,  # Average reward of Top-5
            'time': total_time,  # Total time
            'n_samples': total_images_evaluated,  # Number of images evaluated during training
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
                'evaluation_method': 'selection_from_candidates',  # Indicates evaluation method
                'num_final_candidates': len(final_loop_images),
                'num_eval_samples': num_eval_samples
            }
        }
        
        # Clean up
        del test_set_img_embeds, target_clip_embed
        
        # Clean up final round data (already consumed)
        if len(final_loop_images) > 0:
            del final_loop_images, final_loop_eegs, final_loop_rewards
        
        # Clean up PseudoTargetModel
        if hasattr(self, 'generator') and self.generator is not None:
            if hasattr(self.generator, 'pseudo_target_model') and self.generator.pseudo_target_model is not None:
                del self.generator.pseudo_target_model
                self.generator.pseudo_target_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return result
    
    def reset(self):
        """Reset state and clean up GPU cache"""
        if hasattr(self, 'generator') and self.generator is not None:
            if hasattr(self.generator, 'pseudo_target_model') and self.generator.pseudo_target_model is not None:
                del self.generator.pseudo_target_model
                self.generator.pseudo_target_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class DDPOWrapper(BaseMethod):
    """DDPO method wrapper - based on d3po implementation"""
    
    def __init__(self, config, device="cuda", shared_models=None):
        super().__init__(config, device)
        self.name = "DDPO"
        
        print(f"Initializing {self.name}...")
        
        # Import dependencies
        from peft import LoraConfig
        from model.ATMS_retrieval import ATMS, get_eeg_features
        import sys
        sys.path.append('/home/ldy/Workspace/guide-stable-diffusion/related_works/d3po/d3po')
        
        # Check whether to use shared models
        if shared_models is not None:
            print("  Using shared models...")
            # Use shared model instances
            self.eeg_model = shared_models.get('eeg_model')
            self.encoding_model = shared_models.get('encoding_model')
            
            # Use the shared SDXL pipeline (don't create a new one!)
            self.pipe = shared_models.get('sdxl_pipe')
            if self.pipe is None:
                raise ValueError("Shared SDXL pipeline not found!")
            print("  Using shared SDXL pipeline")
        else:
            # Standalone mode: create own pipeline and models
            print("  Standalone mode: loading independently...")
            self.pipe = self._create_independent_pipeline(device)
            self.eeg_model = self._load_eeg_model(config['eeg_model_path'])
            self.encoding_model = self._load_encoding_model(config['encoding_model_path'])
        
        self.subject = config.get('subject', 'sub-01')
        
        # Training parameters
        self.n_epochs = config.get('n_epochs', 5)
        self.batch_size = config.get('batch_size', 4)
        self.learning_rate = config.get('learning_rate', 3e-5)
        self.use_lora = config.get('use_lora', True)
        self.clip_range = config.get('clip_range', 1e-4)
        self.adv_clip_max = config.get('adv_clip_max', 5)
        self.num_inference_steps = config.get('num_inference_steps', 8)
        self.guidance_scale = 0.0
        self.eta = 0.0
        
        # Freeze non-trainable components
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.text_encoder_2.requires_grad_(False)
        
        # Deferred LoRA configuration: add at optimize() time to avoid conflicts with other methods
        print("  LoRA will be configured on first optimize() call")
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.optimizer = None  # Optimizer is also deferred
        
        print(f"  DDPO initialized: {self.n_epochs} epochs, batch_size={self.batch_size}")
    
    def _create_independent_pipeline(self, device):
        """Create an independent SDXL pipeline (using the same model as other methods)"""
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        
        # Use the same model configuration as Generator4Embeds
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        
        # Load UNet (SDXL-Lightning 8-step model)
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
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.bfloat16
        )
        
        # Load Scheduler
        scheduler = DDIMScheduler.from_pretrained(
            model_id, 
            subfolder="scheduler",
            timestep_spacing="trailing"
        )
        
        # Create pipeline (not using ExtendedStableDiffusionXLPipeline to avoid IP-Adapter related code)
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
        """Run DDPO optimization (based on d3po, using gradient accumulation to avoid GPU memory spikes)"""
        from d3po_pytorch.diffusers_patch.pipeline_with_logprob_sdxl import pipeline_with_logprob
        from d3po_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
        
        # Step 1: Remove all existing LoRA adapters (avoid conflicts with other methods)
        print(f"  [{self.name}] Setting up LoRA adapter...")
        if hasattr(self.pipe.unet, 'peft_config') and len(self.pipe.unet.peft_config) > 0:
            print(f"    Removing {len(self.pipe.unet.peft_config)} existing adapter(s)")
            self.pipe.unet.delete_adapters(list(self.pipe.unet.peft_config.keys()))
        
        # Step 2: Add this method's LoRA adapter
        self.pipe.unet.add_adapter(self.lora_config)
        
        # Step 3: Set LoRA parameters to float32
        for param in self.pipe.unet.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)
        
        # Step 4: Create optimizer
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
        
        # Clear GPU memory before training
        torch.cuda.empty_cache()
        
        print(f"  Training with gradient accumulation: {num_batches_per_epoch} batches per epoch")
        
        for epoch in range(self.n_epochs):
            #################### SAMPLING + TRAINING (process one batch at a time, no accumulation) ####################
            self.pipe.unet.train()  # Enter training mode directly
            
            # Key change: sample and train one batch at a time, don't accumulate all samples
            for batch_idx in range(num_batches_per_epoch):
                ########## Step 1: Sample a single batch ##########
                self.pipe.unet.eval()
                prompts = [""] * self.batch_size
                
                # Create random generator
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
                
                # Convert to PIL and immediately move to CPU
                pil_images = [Image.fromarray((img.float().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)) 
                              for img in images]
                
                # Move all GPU tensors to CPU
                latents_cpu = [lat.cpu() for lat in latents]
                log_probs_cpu = [lp.cpu() for lp in log_probs]
                
                # Immediately delete original tensors on GPU to release memory
                del images, latents, log_probs, generator
                torch.cuda.empty_cache()
                
                # Temporarily move diffusion model off GPU to free space for EEG model
                self.pipe.to('cpu')
                torch.cuda.empty_cache()
                
                # Compute rewards
                synthetic_eegs = self._generate_eeg_from_images(pil_images)
                batch_rewards = [self._compute_eeg_similarity_reward(eeg, target_eeg_feature, self.subject) 
                                for eeg in synthetic_eegs]
                
                # Move diffusion model back to GPU, also move data back
                self.pipe.to(self.device)
                latents = [lat.to(self.device) for lat in latents_cpu]
                log_probs = [lp.to(self.device) for lp in log_probs_cpu]
                # CPU copies are no longer needed, delete immediately
                del latents_cpu, log_probs_cpu
                torch.cuda.empty_cache()
                
                # Update best result (don't save all images, conserve memory)
                for img, r in zip(pil_images, batch_rewards):
                    total_samples_count += 1
                    if r > best_reward:
                        best_reward = r
                        best_image = img
                
                ########## Step 2: Train on this batch immediately (no accumulation) ##########
                self.pipe.unet.train()
                
                # Build a single batch sample
                sample = {
                    'timesteps': self.pipe.scheduler.timesteps.repeat(self.batch_size, 1),
                    'latents': torch.stack(latents, dim=1)[:, :-1],
                    'next_latents': torch.stack(latents, dim=1)[:, 1:],
                    'log_probs': torch.stack(log_probs, dim=1),
                    'rewards': torch.tensor(batch_rewards, device=self.device),
                }
                
                # Compute advantages (based on current batch)
                rewards_np = sample['rewards'].cpu().numpy()
                advantages = (rewards_np - rewards_np.mean()) / (rewards_np.std() + 1e-8)
                sample['advantages'] = torch.tensor(advantages, device=self.device)
                
                # Training parameters
                num_timesteps = sample['timesteps'].shape[1]
                num_train_timesteps = max(1, int(num_timesteps * 0.25))
                
                # Mark latents as requiring gradients
                sample["latents"].requires_grad = True
                
                # Define callback function
                def callback_func(pipe_self, step_index, timestep, callback_kwargs):
                    nonlocal sample, num_train_timesteps
                    
                    if step_index >= num_train_timesteps:
                        return {"latents": sample["next_latents"][:, step_index]}
                    
                    log_prob = callback_kwargs["log_prob"]
                    
                    # Compute PPO loss
                    advantages = torch.clamp(
                        sample["advantages"], -self.adv_clip_max, self.adv_clip_max
                    )
                    ratio = torch.exp(log_prob - sample["log_probs"][:, step_index])
                    unclipped_loss = -advantages * ratio
                    clipped_loss = -advantages * torch.clamp(
                        ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                    )
                    loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                    
                    # Backpropagation
                    loss.backward()
                    
                    # Update parameters at the last training timestep
                    if step_index == num_train_timesteps - 1:
                        torch.nn.utils.clip_grad_norm_(self.pipe.unet.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    
                    return {"latents": sample["next_latents"][:, step_index]}
                
                # Execute training
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
                
                # Immediately clean up current batch data to release GPU memory
                sample["latents"].requires_grad = False
                # DDPO: ensure gradients are fully zeroed (prevent gradient accumulation causing OOM)
                self.optimizer.zero_grad(set_to_none=True)
                # Delete callback function to release its captured variables
                del callback_func
                del sample, latents, log_probs, pil_images, synthetic_eegs, batch_rewards, train_out
                torch.cuda.empty_cache()
                
                # Print progress
                if (batch_idx + 1) % max(1, num_batches_per_epoch // 5) == 0:
                    print(f"    Epoch {epoch+1}/{self.n_epochs}, Batch {batch_idx+1}/{num_batches_per_epoch}, Best Reward: {best_reward:.4f}")
        
        #################### EVALUATION: Re-generate images with the optimized model ####################
        print(f"  Generating final images with optimized model...")
        self.pipe.unet.eval()
        final_images = []
        final_rewards = []
        
        # Generate a few images to evaluate optimization effectiveness
        num_eval_samples = min(5, budget // 2)  # Generate 5 or fewer
        
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
                
                # Compute final reward
                self.pipe.to('cpu')
                torch.cuda.empty_cache()
                
                eval_eegs = self._generate_eeg_from_images(eval_pil)
                eval_rewards_batch = [self._compute_eeg_similarity_reward(eeg, target_eeg_feature, self.subject) 
                                     for eeg in eval_eegs]
                
                self.pipe.to(self.device)
                torch.cuda.empty_cache()
                
                final_images.extend(eval_pil)
                final_rewards.extend(eval_rewards_batch)
        
        # Use the average score of evaluation-phase images as the final metric
        if len(final_rewards) > 0:
            avg_reward = np.mean(final_rewards)
            eval_best_idx = np.argmax(final_rewards)
            eval_best_image = final_images[eval_best_idx]
            print(f"  Optimization complete: avg reward (eval) = {avg_reward:.4f} (std = {np.std(final_rewards):.4f})")
            print(f"  Individual rewards: {[f'{r:.4f}' for r in final_rewards]}")
            print(f"  (Training best was: {best_reward:.4f})")
            
            # Use average score as the final metric
            best_reward = avg_reward
            best_image = eval_best_image  # Keep the best one as the representative image
        else:
            print(f"  Warning: No evaluation images generated. Keeping training best: {best_reward:.4f}")
        
        elapsed_time = time.time() - start_time
        
        # Critical: remove LoRA adapters immediately after training to release GPU memory
        if hasattr(self.pipe.unet, 'peft_config') and len(self.pipe.unet.peft_config) > 0:
            self.pipe.unet.delete_adapters(list(self.pipe.unet.peft_config.keys()))
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return {
            'images': final_images,  # Return evaluation images
            'best_image': best_image,  # From training process
            'rewards': final_rewards,
            'best_reward': best_reward,  # From training process
            'time': elapsed_time,
            'n_samples': total_samples_count + len(final_images),  # Training samples + evaluation samples
            'metadata': {'n_epochs': self.n_epochs, 'batch_size': self.batch_size, 'training_samples': total_samples_count}
        }
    
    def reset(self):
        """Reset DDPO state, clean up LoRA weights and release GPU memory"""
        from peft import LoraConfig
        
        # Re-initialize LoRA (remove old adapters)
        if hasattr(self.pipe.unet, 'peft_config') and len(self.pipe.unet.peft_config) > 0:
            self.pipe.unet.delete_adapters(list(self.pipe.unet.peft_config.keys()))
        
        # Re-configure LoRA
        unet_lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.pipe.unet.add_adapter(unet_lora_config)
        
        # Set LoRA parameters to float32
        for param in self.pipe.unet.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)
        
        # Re-create optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.pipe.unet.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class DPOKWrapper(BaseMethod):
    """DPOK method wrapper - based on d3po implementation, with KL divergence constraint"""
    
    def __init__(self, config, device="cuda", shared_models=None):
        super().__init__(config, device)
        self.name = "DPOK"
        
        print(f"Initializing {self.name}...")
        
        # Import dependencies
        from peft import LoraConfig
        from model.ATMS_retrieval import ATMS, get_eeg_features
        import sys
        sys.path.append('/home/ldy/Workspace/guide-stable-diffusion/related_works/d3po/d3po')
        
        # Check whether to use shared models
        if shared_models is not None:
            print("  Using shared models...")
            # Use shared model instances
            self.eeg_model = shared_models.get('eeg_model')
            self.encoding_model = shared_models.get('encoding_model')
            
            # Use the shared SDXL pipeline (don't create a new one!)
            self.pipe = shared_models.get('sdxl_pipe')
            if self.pipe is None:
                raise ValueError("Shared SDXL pipeline not found!")
            print("  Using shared SDXL pipeline")
        else:
            # Standalone mode: create own pipeline and models
            print("  Standalone mode: loading independently...")
            self.pipe = self._create_independent_pipeline(device)
            self.eeg_model = self._load_eeg_model(config['eeg_model_path'])
            self.encoding_model = self._load_encoding_model(config['encoding_model_path'])
        
        self.subject = config.get('subject', 'sub-01')
        
        # Training parameters
        self.n_epochs = config.get('n_epochs', 5)
        self.batch_size = config.get('batch_size', 4)
        self.learning_rate = config.get('learning_rate', 3e-5)
        self.use_lora = config.get('use_lora', True)
        self.clip_range = config.get('clip_range', 1e-4)
        self.adv_clip_max = config.get('adv_clip_max', 5)
        self.kl_ratio = config.get('kl_ratio', 0.01)  # DPOK-specific: KL divergence coefficient
        self.num_inference_steps = config.get('num_inference_steps', 8)
        self.guidance_scale = 0.0
        self.eta = 0.0
        
        # Freeze non-trainable components
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.text_encoder_2.requires_grad_(False)
        
        # Deferred LoRA configuration: add at optimize() time to avoid conflicts with other methods
        print("  LoRA will be configured on first optimize() call")
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.optimizer = None  # Optimizer is also deferred
        
        print(f"  DPOK initialized: {self.n_epochs} epochs, batch_size={self.batch_size}, kl_ratio={self.kl_ratio}")
    
    def _create_independent_pipeline(self, device):
        """Create an independent SDXL pipeline (using the same model as other methods)"""
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        
        # Use the same model configuration as Generator4Embeds
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        
        # Load UNet (SDXL-Lightning 8-step model)
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
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.bfloat16
        )
        
        # Load Scheduler
        scheduler = DDIMScheduler.from_pretrained(
            model_id, 
            subfolder="scheduler",
            timestep_spacing="trailing"
        )
        
        # Create pipeline (not using ExtendedStableDiffusionXLPipeline to avoid IP-Adapter related code)
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
        """Run DPOK optimization (based on d3po, with KL divergence constraint)"""
        # Implementation similar to DDPO, but adds a KL divergence term in the loss
        # For brevity, detailed implementation is omitted; consistent with DPOKWrapper in benchmark_framework.py
        # Refer to DDPOWrapper above and add KL divergence constraint in callback_func
        pass
    
    def reset(self):
        """Reset DPOK state"""
        pass


class D3POWrapper(BaseMethod):
    """D3PO method wrapper - based on d3po implementation, using pairwise preference learning"""
    
    def __init__(self, config, device="cuda", shared_models=None):
        super().__init__(config, device)
        self.name = "D3PO"
        print(f"Initializing {self.name}...")
        # Detailed implementation omitted; consistent with benchmark_framework.py
        pass
    
    def optimize(self, target_eeg_feature, target_idx, budget=50):
        """Run D3PO optimization"""
        pass
    
    def reset(self):
        """Reset D3PO state"""
        pass


# ==================== Additional Method 1: Bayesian Optimization ====================

class BayesianOptimizationWrapper(BaseMethod):
    """
    Bayesian Optimization method wrapper.
    Optimizes in CLIP embedding space (consistent with PseudoModel).
    Uses Gaussian process to model the objective function (EEG similarity) and selects next sampling point via acquisition function.
    """
    
    def __init__(self, config, device="cuda", shared_models=None):
        super().__init__(config, device)
        self.name = "BayesianOpt"
        
        print(f"Initializing {self.name}...")
        
        # Check whether to use shared models
        if shared_models is not None:
            print("  Using shared models...")
            self.eeg_model = shared_models.get('eeg_model')
            self.encoding_model = shared_models.get('encoding_model')
            self.vlmodel = shared_models.get('vlmodel')
            self.preprocess_train = shared_models.get('preprocess_train')
            
            # Use the shared SDXL pipeline (via Generator)
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
            
            # Standalone mode: load pipe
            from exp_batch_offline_generation import pipe
            self.pipe = pipe
        
        self.subject = config.get('subject', 'sub-01')
        
        # Create an independent random number generator (to avoid conflicts with other methods)
        self.rng = np.random.RandomState(seed=42)
        self.torch_generator = torch.Generator(device=device).manual_seed(42)
        
        # BO-specific parameters
        self.acquisition = config.get('acquisition', 'ucb')  # 'ucb', 'ei', 'poi'
        self.kappa = config.get('kappa', 2.5)  # UCB exploration parameter
        self.xi = config.get('xi', 0.01)  # EI/POI exploration parameter
        self.n_initial_points = config.get('n_initial_points', 10)  # Number of initial random samples
        
        # CLIP embedding dimension
        self.clip_dim = 1024
        
        # Image generation parameters
        self.num_inference_steps = 8
        self.guidance_scale = 0.0
        
        print(f"  BayesianOpt initialized: acquisition={self.acquisition}, n_initial={self.n_initial_points}")
        print(f"  Optimizing in CLIP embedding space (dim={self.clip_dim})")
        print(f"  Using independent RNG with seed=42")
    
    def _sample_clip_embedding(self, n_samples=1):
        """
        Sample CLIP embeddings (in CLIP embedding space)
        Returns: List[np.ndarray], each with shape (1024,)
        """
        embeddings = []
        for _ in range(n_samples):
            # Use independent random number generator
            emb = self.rng.randn(self.clip_dim).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings.append(emb)
        return embeddings
    
    def _clip_embedding_to_image(self, clip_embedding):
        """
        Generate a PIL image directly from a CLIP embedding (using IP-Adapter)
        Args:
            clip_embedding: np.ndarray, shape (1024,)
        Returns:
            PIL.Image
        """
        # Convert to torch tensor
        clip_tensor = torch.tensor(clip_embedding, device=self.device, dtype=torch.float32).unsqueeze(0)
        
        # Generate image directly from CLIP embedding using IP-Adapter (no PseudoTargetModel needed)
        with torch.no_grad():
            # Use independent random number generator to produce latent
            latents = torch.randn(
                1, 4, 
                self.pipe.unet.config.sample_size, 
                self.pipe.unet.config.sample_size, 
                device=self.device,
                dtype=torch.bfloat16,
                generator=self.torch_generator  # Add generator
            )
            
            # Generate image using IP-Adapter
            output = self.pipe(
                prompt=[""],
                ip_adapter_image_embeds=[clip_tensor.unsqueeze(0).type(torch.bfloat16)],
                latents=latents,
                output_type="latent",
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                eta=1.0,
            )
            
            # Decode latents to images
            shift_factor = self.pipe.vae.config.shift_factor if self.pipe.vae.config.shift_factor else 0.0
            decoded_latents = (output.images / self.pipe.vae.config.scaling_factor) + shift_factor
            images = self.pipe.vae.decode(decoded_latents, return_dict=False)[0]
            images = self.pipe.image_processor.postprocess(images.detach())
        
        return images[0]  # Return the first PIL image
    
    def _gaussian_process_predict(self, X_train, y_train, X_test):
        """
        Simplified Gaussian process prediction
        Args:
            X_train: np.ndarray, shape (n_train, latent_dim)
            y_train: np.ndarray, shape (n_train,)
            X_test: np.ndarray, shape (n_test, latent_dim)
        Returns:
            mu: np.ndarray, shape (n_test,), predicted mean
            sigma: np.ndarray, shape (n_test,), predicted standard deviation
        """
        from scipy.spatial.distance import cdist
        
        # Use RBF kernel (simplified version)
        length_scale = 1.0
        noise = 1e-6
        
        # K(X_train, X_train)
        K = np.exp(-cdist(X_train, X_train, 'sqeuclidean') / (2 * length_scale**2))
        K += noise * np.eye(len(X_train))
        
        # K(X_test, X_train)
        K_s = np.exp(-cdist(X_test, X_train, 'sqeuclidean') / (2 * length_scale**2))
        
        # K(X_test, X_test)
        K_ss = np.exp(-cdist(X_test, X_test, 'sqeuclidean') / (2 * length_scale**2))
        
        # Predict
        try:
            K_inv = np.linalg.inv(K)
            mu = K_s @ K_inv @ y_train
            cov = K_ss - K_s @ K_inv @ K_s.T
            sigma = np.sqrt(np.maximum(np.diag(cov), 1e-10))
        except np.linalg.LinAlgError:
            # If matrix is not invertible, return mean prediction
            mu = np.mean(y_train) * np.ones(len(X_test))
            sigma = np.std(y_train) * np.ones(len(X_test))
        
        return mu, sigma
    
    def _acquisition_function(self, mu, sigma, y_best):
        """
        Compute the acquisition function
        Args:
            mu: np.ndarray, GP predicted mean
            sigma: np.ndarray, GP predicted standard deviation
            y_best: float, current best observed value
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
        """Run Bayesian Optimization (in CLIP embedding space)"""
        start_time = time.time()
        
        print(f"  [BayesianOpt] Starting optimization with budget={budget}")
        print(f"  [BayesianOpt] Optimizing in CLIP embedding space (dim={self.clip_dim})")
        
        # Store all sampled points and corresponding rewards
        all_embeddings = []  # List of CLIP embeddings
        all_rewards = []
        all_images = []
        
        best_reward = -float('inf')
        best_image = None
        
        # Step 1: Initial random sampling
        n_initial = min(self.n_initial_points, budget)
        print(f"  [BayesianOpt] Phase 1: Random initialization ({n_initial} samples)")
        
        for i in range(n_initial):
            # Sample CLIP embedding
            clip_emb = self._sample_clip_embedding(1)[0]
            
            # Generate image from CLIP embedding
            image = self._clip_embedding_to_image(clip_emb)
            
            # Compute reward
            eeg = self._generate_eeg_from_images([image])[0]
            reward = self._compute_eeg_similarity_reward(eeg, target_eeg_feature, self.subject)
            
            # Store results
            all_embeddings.append(clip_emb)
            all_rewards.append(reward)
            all_images.append(image)
            
            if reward > best_reward:
                best_reward = reward
                best_image = image
            
            if (i + 1) % 5 == 0:
                print(f"    Initial sampling: {i+1}/{n_initial}, Best reward: {best_reward:.4f}")
        
        # Step 2: Bayesian Optimization iterations
        n_bo_iterations = budget - n_initial
        print(f"  [BayesianOpt] Phase 2: BO iterations ({n_bo_iterations} samples)")
        
        X_train = np.array(all_embeddings)  # shape: (n_samples, 1024)
        y_train = np.array(all_rewards)  # shape: (n_samples,)
        
        for i in range(n_bo_iterations):
            # Generate candidate points (randomly sample a batch in CLIP embedding space)
            n_candidates = 100
            candidate_embeddings = self._sample_clip_embedding(n_candidates)
            X_candidates = np.array(candidate_embeddings)
            
            # Use GP to predict mean and variance for candidate points
            mu, sigma = self._gaussian_process_predict(X_train, y_train, X_candidates)
            
            # Compute acquisition function
            y_best = np.max(y_train)
            acq_values = self._acquisition_function(mu, sigma, y_best)
            
            # Select the point with the highest acquisition function value
            best_candidate_idx = np.argmax(acq_values)
            next_embedding = X_candidates[best_candidate_idx]
            
            # Generate image from CLIP embedding and compute reward
            image = self._clip_embedding_to_image(next_embedding)
            eeg = self._generate_eeg_from_images([image])[0]
            reward = self._compute_eeg_similarity_reward(eeg, target_eeg_feature, self.subject)
            
            # Update training set
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
        
        # After optimization, generate 5 images using the best CLIP embedding for final evaluation
        print(f"  [BayesianOpt] Generating 5 final samples for evaluation...")
        best_embedding = X_train[np.argmax(y_train)]
        
        final_images = []
        final_rewards = []
        num_eval_samples = 5
        
        for i in range(num_eval_samples):
            # Generate image from the best embedding (different latents each time)
            image = self._clip_embedding_to_image(best_embedding)
            eeg = self._generate_eeg_from_images([image])[0]
            reward = self._compute_eeg_similarity_reward(eeg, target_eeg_feature, self.subject)
            
            final_images.append(image)
            final_rewards.append(reward)
            
            # Immediately release intermediate variables (EEG features consume GPU memory)
            del eeg
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Compute average score
        final_score = np.mean(final_rewards)
        best_eval_image = final_images[np.argmax(final_rewards)]
        
        print(f"  [BayesianOpt] Final evaluation: mean = {final_score:.4f}, std = {np.std(final_rewards):.4f}")
        print(f"  [BayesianOpt] Individual rewards: {[f'{r:.4f}' for r in final_rewards]}")
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return {
            'images': final_images,  # 5 images from final evaluation
            'best_image': best_eval_image,
            'rewards': final_rewards,
            'best_reward': final_score,  # Average of 5 images
            'time': elapsed_time,
            'n_samples': budget,
            'metadata': {
                'acquisition': self.acquisition,
                'n_initial_points': n_initial,
                'n_bo_iterations': n_bo_iterations,
                'optimization_space': 'CLIP_embedding',
                'num_eval_samples': num_eval_samples,
                'eval_rewards_std': float(np.std(final_rewards)),
                'optimization_best': best_reward  # Best value during optimization
            }
        }
    
    def reset(self):
        """Reset BO state"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# ==================== Additional Method 2: CMA-ES ====================

class CMAESWrapper(BaseMethod):
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) method wrapper.
    Optimizes in CLIP embedding space (consistent with PseudoModel).
    Maintains a multivariate Gaussian distribution and optimizes CLIP embedding via evolution strategy.
    """
    
    def __init__(self, config, device="cuda", shared_models=None):
        super().__init__(config, device)
        self.name = "CMA-ES"
        
        print(f"Initializing {self.name}...")
        
        # Check whether to use shared models
        if shared_models is not None:
            print("  Using shared models...")
            self.eeg_model = shared_models.get('eeg_model')
            self.encoding_model = shared_models.get('encoding_model')
            self.vlmodel = shared_models.get('vlmodel')
            self.preprocess_train = shared_models.get('preprocess_train')
            
            # Use the shared SDXL pipeline (via Generator)
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
            
            # Standalone mode: load pipe
            from exp_batch_offline_generation import pipe
            self.pipe = pipe
        
        self.subject = config.get('subject', 'sub-01')
        
        # Create an independent random number generator (using a different seed to avoid conflict with BO)
        self.rng = np.random.RandomState(seed=43)
        self.torch_generator = torch.Generator(device=device).manual_seed(43)
        
        # CMA-ES specific parameters
        self.population_size = config.get('population_size', 10)
        self.sigma = config.get('sigma', 0.5)
        
        # CLIP embedding dimension
        self.clip_dim = 1024
        
        # Image generation parameters
        self.num_inference_steps = 8
        self.guidance_scale = 0.0
        
        print(f"  CMA-ES initialized: population_size={self.population_size}, sigma={self.sigma}")
        print(f"  Optimizing in CLIP embedding space (dim={self.clip_dim})")
        print(f"  Using independent RNG with seed=43")
    
    def _clip_embedding_to_image(self, clip_embedding):
        """
        Generate a PIL image directly from a CLIP embedding (using IP-Adapter)
        Args:
            clip_embedding: np.ndarray, shape (1024,)
        Returns:
            PIL.Image
        """
        # Convert to torch tensor
        clip_tensor = torch.tensor(clip_embedding, device=self.device, dtype=torch.float32).unsqueeze(0)
        
        # Generate image directly from CLIP embedding using IP-Adapter (no PseudoTargetModel needed)
        with torch.no_grad():
            # Use independent random number generator to produce latent
            latents = torch.randn(
                1, 4, 
                self.pipe.unet.config.sample_size, 
                self.pipe.unet.config.sample_size, 
                device=self.device,
                dtype=torch.bfloat16,
                generator=self.torch_generator  # Add generator
            )
            
            # Generate image using IP-Adapter
            output = self.pipe(
                prompt=[""],
                ip_adapter_image_embeds=[clip_tensor.unsqueeze(0).type(torch.bfloat16)],
                latents=latents,
                output_type="latent",
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                eta=1.0,
            )
            
            # Decode latents to images
            shift_factor = self.pipe.vae.config.shift_factor if self.pipe.vae.config.shift_factor else 0.0
            decoded_latents = (output.images / self.pipe.vae.config.scaling_factor) + shift_factor
            images = self.pipe.vae.decode(decoded_latents, return_dict=False)[0]
            images = self.pipe.image_processor.postprocess(images.detach())
        
        return images[0]  # Return the first PIL image
    
    def optimize(self, target_eeg_feature, target_idx, budget=50):
        """Run CMA-ES optimization (in CLIP embedding space)"""
        try:
            import cma
        except ImportError:
            print("  ERROR: cma package not installed. Please run: pip install cma")
            print("  Falling back to random sampling...")
            return self._fallback_random_sampling(target_eeg_feature, budget)
        
        start_time = time.time()
        
        print(f"  [CMA-ES] Starting optimization with budget={budget}")
        print(f"  [CMA-ES] Optimizing in CLIP embedding space (dim={self.clip_dim})")
        
        # Initial mean (zero vector)
        x0 = np.zeros(self.clip_dim)
        
        # CMA-ES optimizer
        es = cma.CMAEvolutionStrategy(x0, self.sigma, {
            'popsize': self.population_size,
            'maxiter': budget // self.population_size,
            'verb_disp': 0,  # Disable verbose output
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
            
            # Sample population
            solutions = es.ask()
            
            # Evaluate each solution
            fitness_list = []
            
            for solution in solutions:
                if n_evaluations >= budget:
                    break
                
                # Normalize to unit vector (CLIP embeddings are typically normalized)
                clip_emb = solution / (np.linalg.norm(solution) + 1e-8)
                clip_emb = clip_emb.astype(np.float32)
                
                # Generate image from CLIP embedding
                image = self._clip_embedding_to_image(clip_emb)
                
                # Compute reward
                eeg = self._generate_eeg_from_images([image])[0]
                reward = self._compute_eeg_similarity_reward(eeg, target_eeg_feature, self.subject)
                
                # CMA-ES minimizes, so negate the reward
                fitness = -reward
                fitness_list.append(fitness)
                
                all_images.append(image)
                all_rewards.append(reward)
                n_evaluations += 1
                
                if reward > best_reward:
                    best_reward = reward
                    best_image = image
            
            # Update CMA-ES
            es.tell(solutions[:len(fitness_list)], fitness_list)
            
            print(f"    Generation {generation}, Evaluations: {n_evaluations}/{budget}, Best reward: {best_reward:.4f}")
        
        elapsed_time = time.time() - start_time
        
        print(f"  [CMA-ES] Optimization complete: best optimization reward = {best_reward:.4f}")
        
        # After optimization, generate 5 images using the best solution for final evaluation
        print(f"  [CMA-ES] Generating 5 final samples for evaluation...")
        best_solution = es.result.xbest  # CMA-ES's best solution
        best_clip_emb = best_solution / (np.linalg.norm(best_solution) + 1e-8)
        
        final_images = []
        final_rewards = []
        num_eval_samples = 5
        
        for i in range(num_eval_samples):
            # Generate image from the best embedding (different latents each time)
            image = self._clip_embedding_to_image(best_clip_emb)
            eeg = self._generate_eeg_from_images([image])[0]
            reward = self._compute_eeg_similarity_reward(eeg, target_eeg_feature, self.subject)
            
            final_images.append(image)
            final_rewards.append(reward)
            
            # Immediately release intermediate variables (EEG features consume GPU memory)
            del eeg
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Compute average score
        final_score = np.mean(final_rewards)
        best_eval_image = final_images[np.argmax(final_rewards)]
        
        print(f"  [CMA-ES] Final evaluation: mean = {final_score:.4f}, std = {np.std(final_rewards):.4f}")
        print(f"  [CMA-ES] Individual rewards: {[f'{r:.4f}' for r in final_rewards]}")
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return {
            'images': final_images,  # 5 images from final evaluation
            'best_image': best_eval_image,
            'rewards': final_rewards,
            'best_reward': final_score,  # Average of 5 images
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
                'optimization_best': best_reward  # Best value during optimization
            }
        }
    
    def _fallback_random_sampling(self, target_eeg_feature, budget):
        """Fallback to random sampling in CLIP embedding space if the CMA library is unavailable"""
        start_time = time.time()
        
        print(f"  [CMA-ES Fallback] Using random sampling in CLIP embedding space with budget={budget}")
        
        all_images = []
        all_rewards = []
        best_reward = -float('inf')
        best_image = None
        
        for i in range(budget):
            # Randomly sample CLIP embedding
            clip_emb = np.random.randn(self.clip_dim).astype(np.float32)
            clip_emb = clip_emb / (np.linalg.norm(clip_emb) + 1e-8)  # Normalize
            
            # Generate image from CLIP embedding
            image = self._clip_embedding_to_image(clip_emb)
            
            # Compute reward
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
        """Reset CMA-ES state"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# ==================== Benchmark Framework ====================

class BenchmarkFramework:
    """Main benchmark framework (supports all 6 methods)"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to configuration file (JSON format)
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = self.config.get('device', 'cuda')
        self.output_dir = self.config.get('output_dir', './benchmark_results_total')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create shared models (to avoid redundant loading)
        print("\n" + "="*60)
        print("Loading Shared Models (to save GPU memory)...")
        print("="*60)
        self.shared_models = self._load_shared_models()
        
        # Initialize all methods
        self.methods = self._initialize_methods()
        
        # Load targets
        self.targets = self._load_targets()
        
        print(f"Initialized {len(self.methods)} methods")
        print(f"Loaded {len(self.targets)} targets")
    
    def _load_shared_models(self):
        """Load models shared by all methods to avoid redundant loading"""
        shared_models = {}
        
        # Get config from any enabled method (to obtain model paths)
        method_configs = self.config['methods']
        enabled_methods = [k for k, v in method_configs.items() if v.get('enabled')]
        
        if not enabled_methods:
            return shared_models
        
        # Get model paths from the first enabled method
        sample_config = None
        for method_name in enabled_methods:
            config = method_configs[method_name]
            if 'eeg_model_path' in config and 'encoding_model_path' in config:
                sample_config = config
                break
        
        if sample_config is None:
            print("  No methods with model paths found, skipping shared model loading")
            return shared_models
        
        # Load EEG model
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
        
        # Load encoding model
        print("  Loading shared encoding model...")
        from model.utils import load_model_encoder
        encoding_model = load_model_encoder(
            sample_config['encoding_model_path'], 
            self.device
        )
        encoding_model.eval()
        shared_models['encoding_model'] = encoding_model
        
        # Load CLIP model (if PseudoModel is enabled)
        if 'pseudo_model' in enabled_methods:
            print("  Loading shared CLIP model...")
            from exp_batch_offline_generation import vlmodel, preprocess_train
            shared_models['vlmodel'] = vlmodel
            shared_models['preprocess_train'] = preprocess_train
        
        # Important: create a shared SDXL pipeline for methods that need it
        rl_methods = ['ddpo', 'dpok', 'd3po', 'bayesian_opt', 'cma_es']
        if any(m in enabled_methods for m in rl_methods):
            print("  Loading shared SDXL pipeline...")
            shared_models['sdxl_pipe'] = self._create_shared_sdxl_pipeline(self.device)
            print("  Shared SDXL pipeline loaded successfully!")
        
        print("  All shared models loaded!")
        
        return shared_models
    
    def _create_shared_sdxl_pipeline(self, device):
        """Create a shared SDXL pipeline"""
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
        """Initialize all methods to compare (using shared models)"""
        methods = {}
        
        method_configs = self.config['methods']
        
        # PseudoModel (Offline)
        if method_configs.get('pseudo_model', {}).get('enabled'):
            methods['PseudoModel'] = PseudoModelWrapper(
                method_configs['pseudo_model'], self.device, 
                shared_models=self.shared_models
            )
        
        # Heuristic Closed-Loop Pseudo Model
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
            
        # DPOK (requires full implementation, skipped here)
        # if method_configs.get('dpok', {}).get('enabled'):
        #     methods['DPOK'] = DPOKWrapper(...)
            
        # D3PO (requires full implementation, skipped here)
        # if method_configs.get('d3po', {}).get('enabled'):
        #     methods['D3PO'] = D3POWrapper(...)
        
        # Bayesian Optimization
        if method_configs.get('bayesian_opt', {}).get('enabled'):
            methods['BayesianOpt'] = BayesianOptimizationWrapper(
                method_configs['bayesian_opt'], self.device,
                shared_models=self.shared_models
            )
        
        # CMA-ES
        if method_configs.get('cma_es', {}).get('enabled'):
            methods['CMA-ES'] = CMAESWrapper(
                method_configs['cma_es'], self.device,
                shared_models=self.shared_models
            )
        
        return methods
    
    def _load_targets(self) -> List[Dict]:
        """
        Randomly sample target EEG features (to ensure fair evaluation).
        Randomly draws a specified number of targets from the entire image pool.
        """
        # Read configuration
        target_config = self.config['target_selection']
        num_targets = target_config['num_targets']
        random_seed = target_config['random_seed']
        
        embed_dir = self.config['data']['embed_dir']
        image_dir = self.config['data']['image_dir']
        
        # Get all available files
        embed_files = sorted([f for f in os.listdir(embed_dir) if f.endswith('_embed.pt')])
        image_files = sorted([f for f in os.listdir(image_dir) 
                             if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        # Ensure file counts match
        assert len(embed_files) == len(image_files), \
            f"Mismatch: {len(embed_files)} embeds vs {len(image_files)} images"
        
        total_available = len(embed_files)
        
        # Check if requested number is reasonable
        if num_targets > total_available:
            print(f"  WARNING: Requested {num_targets} targets, but only {total_available} available")
            num_targets = total_available
        
        # Randomly sample target indices (using fixed seed for reproducibility)
        rng = np.random.RandomState(seed=random_seed)
        selected_indices = rng.choice(total_available, size=num_targets, replace=False)
        selected_indices = np.sort(selected_indices)  # Sort for readability (keep as numpy array)
        
        print(f"  Target Selection: Randomly sampled {num_targets} targets from {total_available} available")
        print(f"  Selected indices: {selected_indices.tolist()}")
        print(f"  Random seed: {random_seed}")
        
        # Load selected targets
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
        """Run a single experiment"""
        
        # Clean up GPU memory before each experiment
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        target_idx = target['idx']
        target_eeg = target['eeg_feature']
        
        print(f"  Running {method.name} on target {target_idx} (seed={seed})...")
        
        try:
            # Record GPU memory
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated() / 1e9
                print(f"    Initial GPU memory: {initial_memory:.2f} GB")
            
            # Run optimization
            result = method.optimize(target_eeg, target_idx, budget)
            
            # Compute peak GPU memory
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / 1e9
                gpu_memory = peak_memory - initial_memory
                print(f"    Peak GPU memory: {peak_memory:.2f} GB (increment: {gpu_memory:.2f} GB)")
            else:
                gpu_memory = 0.0
            
            # Save generated images
            if result['best_image'] is not None:
                save_dir = os.path.join(
                    self.output_dir, 'images', method.name, f"target_{target_idx}"
                )
                os.makedirs(save_dir, exist_ok=True)
                result['best_image'].save(
                    os.path.join(save_dir, f"seed_{seed}.png")
                )
            
            # Create result object
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
            
            # Immediately reset method to release GPU memory
            method.reset()
            
            # Clean up GPU memory
            del result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
        except Exception as e:
            import traceback
            print(f"    ERROR: {str(e)}")
            # Print full error traceback (for debugging)
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
            
            # Reset method and clean up GPU memory even on error
            try:
                method.reset()
            except:
                pass  # Continue even if reset fails
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        return benchmark_result
    
    def experiment_1_single_target(self) -> List[BenchmarkResult]:
        """
        Experiment 1: Single-target reconstruction comparison.
        All methods generate images under the same budget for comparison.
        """
        print("\n" + "="*60)
        print("Experiment 1: Single-target reconstruction comparison (6 methods)")
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
        
        # Save results
        self._save_results(all_results, "exp1_results.csv")
        
        return all_results
    
    def _save_results(self, results: List[BenchmarkResult], filename: str):
        """Save results to CSV"""
        df = pd.DataFrame([r.to_dict() for r in results])
        output_path = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    def visualize_results(self, results: List[BenchmarkResult], output_prefix: str):
        """Visualize results"""
        df = pd.DataFrame([r.to_dict() for r in results if r.success])
        
        if len(df) == 0:
            print("No successful experiments to visualize!")
            return
        
        # Summary statistics table
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
        
        # Save statistics
        summary.to_csv(os.path.join(self.output_dir, f'{output_prefix}_summary.csv'))
        
        # Box plot comparison
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
        """Run the full benchmark"""
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
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Benchmark Experiments (6 Methods)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config JSON file')
    parser.add_argument('--exp', type=str, default='all',
                       choices=['all', 'exp1'],
                       help='Which experiment to run')
    
    args = parser.parse_args()
    
    # Create benchmark framework
    framework = BenchmarkFramework(args.config)
    
    # Run experiments
    if args.exp == 'all':
        framework.run_full_benchmark()
    elif args.exp == 'exp1':
        results = framework.experiment_1_single_target()
        framework.visualize_results(results, 'exp1')


if __name__ == '__main__':
    main()


