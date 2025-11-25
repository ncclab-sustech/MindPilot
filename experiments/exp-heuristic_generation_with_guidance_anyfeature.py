import os

import numpy as np
import torch
import sys
import random
from PIL import Image
from scipy.special import softmax
import open_clip
from mne.time_frequency import psd_array_multitaper
import torch.nn.functional as F
import torch.nn as nn
from scipy.special import softmax
from datetime import datetime
sys.path.append('/home/ldy/Workspace/Closed_loop_optimizing')
sys.path.append('/home/ldy/Workspace/Closed_loop_optimizing/model')
# from torchvision import transforms
from torchvision import models
from model.utils import load_model_encoder, generate_eeg, save_eeg_signal
import matplotlib.pyplot as plt
from model.custom_pipeline_low_level import Generator4Embeds
from IPython.display import display
import torchvision.transforms as transforms
from model.ATMS_retrieval import ATMS, get_eeg_features

import numpy as np
from util import save_eeg, get_gteeg
import einops
from model.pseudo_target_model import PseudoTargetModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import logging
logging.getLogger("diffusers").setLevel(logging.WARNING)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_type = 'ViT-H-14'

vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
    model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device = device)
vlmodel.to(device)


generator = Generator4Embeds(device=device)

pipe = generator.pipe


# 配置图片和embed目录
image_dir = '/home/ldy/Workspace/Closed_loop_optimizing/test_images'
embed_dir = '/home/ldy/Workspace/Closed_loop_optimizing/data/clip_embed/2025-09-21'

# 获取所有图片和embed路径，排序保证一一对应
image_list = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
embed_list = sorted([f for f in os.listdir(embed_dir) if f.endswith('_embed.pt')])



def main_loop(target_idx):
    assert target_idx < len(image_list), f"target_idx超出图片数量范围({len(image_list)})"
    assert target_idx < len(embed_list), f"target_idx超出embed数量范围({len(embed_list)})"

    target_image_path = os.path.join(image_dir, image_list[target_idx])
    target_eeg_embed_path = os.path.join(embed_dir, embed_list[target_idx])
    print(f"当前索引: {target_idx}")
    print(f"图片: {target_image_path}")
    print(f"eeg_embed: {target_eeg_embed_path}")

    # 其余参数保持不变
    sub = 'sub-01'
    fs = 250
    selected_channel_idxes = slice(None)
    random.seed(43)
    dnn = 'alexnet'
    encoding_model_path = f'/home/ldy/Workspace/Closed_loop_optimizing/kyw/closed-loop/EEG-encoding/EEG_encoder/results/{sub}/synthetic_eeg_data/encoding-end_to_end/dnn-{dnn}/modeled_time_points-all/pretrained-True/lr-1e-05__wd-0e+00__bs-064/model_state_dict.pt'
    target_feature = torch.load(target_eeg_embed_path)
    f_encoder =  "/home/ldy/Workspace/Closed_loop_optimizing/kyw/closed-loop/sub_model/sub-01/diffusion_alexnet/pretrained_True/gene_gene/ATM_S_reconstruction_scale_0_1000_40.pth"
    checkpoint = torch.load(f_encoder, map_location=device)
    eeg_model = ATMS()
    eeg_model.load_state_dict(checkpoint['eeg_model_state_dict'])

    save_path = f"/home/ldy/Workspace/Closed_loop_optimizing/outputs/iclr2026"
    os.makedirs(save_path, exist_ok=True)

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


    def load_target_feature(encoding_model_path, target_image_path, fs, selected_channel_idxes, device):
        target_signal = generate_eeg_from_image_paths(encoding_model_path, [target_image_path], device=device)

        # if selected_channel_idxes==slice(None):        
        #     selected_target_signal = target_signal[:, :]
        # else:
        selected_target_signal = target_signal[selected_channel_idxes, :]                
        target_psd, _ = psd_array_multitaper(selected_target_signal, fs, adaptive=True, normalization='full', verbose=0)
        return torch.from_numpy(target_psd.flatten()).unsqueeze(0)

    def get_image_pool(image_set_path):
        test_images_path = []
        for root, dirs, files in os.walk(image_set_path):
            for file in sorted(files):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    test_images_path.append(os.path.join(root, file))
        return test_images_path

    test_images_path = get_image_pool(image_dir)
    print(f"test_images_path {test_images_path}")

    if target_image_path in test_images_path:
        test_images_path.remove(target_image_path)

    len(test_images_path)

    def generate_eeg_from_image(model_path, images, device):
        synthetic_eegs = []
        model = load_model_encoder(model_path, device)
        for idx, image in enumerate(images):
            image_tensor = preprocess_generated_image(image, device)
            synthetic_eeg = generate_eeg(model, image_tensor, device)
            synthetic_eegs.append(synthetic_eeg)
            # category = category_list[idx]
            # save_eeg_signal(synthetic_eeg, save_dir, idx, category)
        return synthetic_eegs

    def preprocess_generated_image(image, device):
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])    

        image_tensor = transform(image).unsqueeze(0).to(device)
        return image_tensor

    def calculate_loss_from_eeg_path(eeg_path, target_feature, fs, selected_channel_idxes):
        # eeg = np.load(eeg_path, allow_pickle=True)
        # selected_eeg = eeg[selected_channel_idxes, :]
        # psd, _ = psd_array_multitaper(selected_eeg, fs, adaptive=True, normalization='full', verbose=0)
        # psd = torch.from_numpy(psd.flatten()).unsqueeze(0)
        # target_feature = torch.tensor(target_feature).view(1, 378)
        # loss_fn = nn.MSELoss()
        # loss = loss_fn(psd, target_feature)    
        loss = 0
        return loss

    def calculate_loss(eeg, target_feature, fs, selected_channel_idxes):    
        # selected_eeg = eeg[selected_channel_idxes, :]
        # psd, _ = psd_array_multitaper(selected_eeg, fs, adaptive=True, normalization='full', verbose=0)
        # psd = torch.from_numpy(psd.flatten()).unsqueeze(0)
        # target_feature = torch.tensor(target_feature).view(1, 378)
        # loss_fn = nn.MSELoss()
        # loss = loss_fn(psd, target_feature)
        loss = 0
        return loss

    def calculate_loss_clip_embed():
        loss = 0
        return loss

    def calculate_loss_clip_embed_image():
        loss = 0
        return loss

    def reward_function_from_eeg_path(eeg_path, target_feature, fs, selected_channel_idxes):
        eeg = np.load(eeg_path, allow_pickle=True)
        selected_eeg = eeg[selected_channel_idxes, :]
        psd, _ = psd_array_multitaper(selected_eeg, fs, adaptive=True, normalization='full', verbose=0)
        psd = torch.from_numpy(psd.flatten()).unsqueeze(0)
        return F.cosine_similarity(target_feature, psd).item()


    def load_psd_from_eeg(target_signal, fs, selected_channel_idxes):
        selected_target_signal = target_signal[selected_channel_idxes, :]
        psd_feature, _ = psd_array_multitaper(selected_target_signal, fs, adaptive=True, normalization='full', verbose=0)
        return torch.from_numpy(psd_feature.flatten()).unsqueeze(0)

    def reward_function(eeg, target_feature, fs, selected_channel_idxes):    
        selected_eeg = eeg[selected_channel_idxes, :]
        psd, _ = psd_array_multitaper(selected_eeg, fs, adaptive=True, normalization='full', verbose=0)
        psd = torch.from_numpy(psd.flatten()).unsqueeze(0)
        return F.cosine_similarity(target_feature, psd).item()



    def reward_function_clip_embed_image(pil_image, target_feature):
        """
        生成与某张图片对应的脑电信号，并与 groundtruth 进行相似度计算
        :param image: 图片特征向量 [1024]
        :param groundtruth_eeg: groundtruth 的特征向量 [1024]
        :return: EEG信号与groundtruth的相似度
        """    
        tensor_images = [preprocess_train(pil_image)]    
        with torch.no_grad():
            img_embeds = vlmodel.encode_image(torch.stack(tensor_images).to(device))      

        similarity = torch.nn.functional.cosine_similarity(img_embeds.to(device), target_feature.to(device))

        similarity = (similarity + 1) / 2    
        # print(similarity)
        return similarity.item()

    def reward_function_clip_embed(eeg, eeg_model, target_feature, sub, dnn):
        """
        生成与某张图片对应的脑电信号，并与 groundtruth 进行相似度计算
        :param image: 图片特征向量 [1024]
        :param groundtruth_eeg: groundtruth 的特征向量 [1024]
        :return: EEG信号与groundtruth的相似度
        """    
        eeg_feature = get_eeg_features(eeg_model, torch.tensor(eeg).unsqueeze(0), device, sub)    
        similarity = torch.nn.functional.cosine_similarity(eeg_feature.to(device), target_feature.to(device))
        # cos_sim = F.softmax(cos_sim)
        similarity = (similarity + 1) / 2

        # print(similarity)
        return similarity.item(), eeg_feature

    class HeuristicGenerator:
        def __init__(self, pipe, vlmodel, preprocess_train, device="cuda"):
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

            # Initialize components
            self.pseudo_target_model = PseudoTargetModel(dimension=self.dimension, noise_level=1e-4).to(self.device)
            self.generator = torch.Generator(device=device).manual_seed(0)

            # Load IP adapter
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter", subfolder="sdxl_models", 
                weight_name="ip-adapter_sdxl_vit-h.bin", 
                torch_dtype=torch.bfloat16)
            self.pipe.set_ip_adapter_scale(0.5)

        def reward_function_embed(self, embed1, embed2):
            """
            Compute reward based on cosine similarity between CLIP embeddings

            Args:
                embed1: First set of embeddings (batch_size, embedding_dim)
                embed2: Second set of embeddings (batch_size, embedding_dim)

            Returns:
                Normalized similarity scores in [0, 1] range
            """
            # Compute cosine similarity (range [-1, 1])
            cosine_sim = F.cosine_similarity(embed1, embed2, dim=1)

            # Normalize to [0, 1] range
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
            # Add model data
            # print(f"data_x {data_x[0].shape}")
            # print(f"data_y {data_y}")

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

            for step in range(self.total_steps):
                # Generate latents and images
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

                images = self.latents_to_images(latents)         

                # 新增：保存每步生成的图片
                if img_save_dir is not None:
                    for img in images:
                        img.save(os.path.join(img_save_dir, f"img_{img_counter:03d}.png"))
                        img_counter += 1

                data_x, data_y = self.pseudo_target_model.get_model_data()   
                if data_y.size(0) < 50: #w/o optimization 
                    return images

                image_inputs = torch.stack([self.preprocess_train(img) for img in images])

                with torch.no_grad():
                    image_features = self.vlmodel.encode_image(image_inputs.to(self.device)) 

                step_size = self.initial_step_size / (1 + self.decay_rate * step)
                pseudo_target, _ = self.pseudo_target_model.estimate_pseudo_target(image_features, step_size=step_size) #batchsize, hidden_dim

                if step % self.save_per == 0:
                    all_images.append(images)
                del latents, images, image_inputs, image_features
                torch.cuda.empty_cache()

            if plots_save_folder is not None:
                merged_image = self.merge_images_grid(all_images)
                merged_image.save(os.path.join(plots_save_folder, "merged.png"))

            return self.latents_to_images(self.pipe(
                    [prompt]*self.generate_batch_size,
                    ip_adapter_image_embeds=[pseudo_target.unsqueeze(0).type(torch.bfloat16).to(self.device)],
                    latents=epsilon[0].type(torch.bfloat16),
                    given_noise=epsilon[1:].type(torch.bfloat16),
                    output_type="latent",
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    eta=1.0,
                ).images)


    def fusion_image_to_images(Generator, img_embeds, rewards, device, scale, save_path=None):        
            # 随机选择两个不同的索引
        idx1, idx2 = random.sample(range(len(img_embeds)), 2)
        # 获取对应的嵌入向量并添加批次维度
        embed1, embed2 = img_embeds[idx1].unsqueeze(0), img_embeds[idx2].unsqueeze(0)
        embed_len = embed1.size(1)
        start_idx = random.randint(0, embed_len - scale - 1)
        end_idx = start_idx + scale
        temp = embed1[:, start_idx:end_idx].clone()
        embed1[:, start_idx:end_idx] = embed2[:, start_idx:end_idx]
        embed2[:, start_idx:end_idx] = temp
        # print(f"chosen_images {len(chosen_images)}")
        # print(f"rewards {len(rewards)}")
        generated_images = []        
        with torch.no_grad():         
            images = Generator.generate(img_embeds.to(device), torch.tensor(rewards).to(device), target_feature, prompt='', save_path=save_path, start_embedding=embed1)
            # image = generator.generate(embed1)
            generated_images.extend(images)
            # print(f"type(images) {type(images)}")
            images = Generator.generate(img_embeds.to(device), torch.tensor(rewards).to(device), target_feature, prompt='', save_path=save_path, start_embedding=embed2)
            # image = generator.generate(embed2)
            generated_images.extend(images)

        return generated_images


    def select_from_image_paths(probabilities, similarities, losses, sample_image_paths, synthetic_eegs, device, size):
        chosen_indices = np.random.choice(len(probabilities), size=size, replace=False, p=probabilities)
        # print(f"sample_image_paths {len(sample_image_paths)}")
        # print(f"chosen_indices  {chosen_indices}")

        chosen_similarities = [similarities[idx] for idx in chosen_indices.tolist()] 
        chosen_losses = [losses[idx] for idx in chosen_indices.tolist()]    
        chosen_images = [Image.open(sample_image_paths[i]).convert("RGB") for i in chosen_indices.tolist()]        
        chosen_eegs = [synthetic_eegs[idx] for idx in chosen_indices.tolist()]
        return chosen_similarities, chosen_losses, chosen_images, chosen_eegs


    def select_from_images(probabilities, similarities, losses, images_list, eeg_list, size):
        chosen_indices = np.random.choice(len(similarities), size=size, replace=False, p=probabilities)
        # print(f"eeg_list {len(eeg_list)}")
        # print(f"chosen_indices  {chosen_indices}")    
        chosen_similarities = [similarities[idx] for idx in chosen_indices.tolist()] 
        chosen_losses = [losses[idx] for idx in chosen_indices.tolist()]
        chosen_images = [images_list[idx] for idx in chosen_indices.tolist()]
        chosen_eegs = [eeg_list[idx] for idx in chosen_indices.tolist()]
        return chosen_similarities, chosen_losses, chosen_images, chosen_eegs

    def get_prob_random_sample(test_images_path, model_path, fs, device, selected_channel_idxes, processed_paths, target_feature, size):
        available_paths = [path for path in test_images_path if path not in processed_paths]    
        sample_image_paths = sorted(random.sample(available_paths, 10))
        # print(f"sample_image_paths {sample_image_paths}")
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
            cs, eeg_feature = reward_function_clip_embed(eeg, eeg_model, target_feature, sub, dnn)
            loss = calculate_loss_clip_embed()
            similarities.append(cs)
            losses.append(loss)

        probabilities = softmax(similarities)


        chosen_similarities, chosen_losses, chosen_images, chosen_eegs = select_from_image_paths(probabilities, similarities, losses, sample_image_paths, synthetic_eegs, device, size)

        return chosen_similarities, chosen_losses, chosen_images, chosen_eegs

    def visualize_top_images(images, similarities, save_folder, iteration):
        """
        使用 matplotlib 按相似度顺序显示选中的图片
        :param image_paths: 图片路径列表
        :param similarities: 每张图片的相似度列表
        """
        # 将图片路径和相似度结合，并按相似度降序排序
        image_similarity_pairs = sorted(zip(images, similarities), key=lambda x: x[1], reverse=True)

        # 拆分排序后的图片路径和相似度
        sorted_images, sorted_similarities = zip(*image_similarity_pairs)

        # 绘制图像
        fig, axes = plt.subplots(1, len(sorted_images), figsize=(15, 5))
        for i, image in enumerate(sorted_images):
            axes[i].imshow(image)
            axes[i].axis('off')
            axes[i].set_title(f'Similarity: {sorted_similarities[i]:.4f}', fontsize=8)  # 显示相似度
        plt.show()

        os.makedirs(save_folder, exist_ok=True)  # 创建文件夹（如果不存在）
        save_path = os.path.join(save_folder, f"visualization_iteration_{iteration}.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)  # 保存图像文件
        print(f"Visualization saved to {save_path}")

    def compute_embed_similarity(img_feature, all_features):
        """
        计算某张图片与所有其他图片的余弦相似度（结果在0-1之间）
        :param img_feature: 选中图片的特征向量 [D] 或 [1, D]
        :param all_features: 所有图片的特征向量 [N, D]
        :return: 余弦相似度 [N] (范围0-1)
        """
        # 确保输入是浮点类型
        img_feature = img_feature.float()
        all_features = all_features.float()

        # 确保特征向量是2D的 [1, D]
        if img_feature.dim() == 1:
            img_feature = img_feature.unsqueeze(0)

        # 检查NaN/Inf值
        assert torch.isfinite(img_feature).all(), "img_feature contains NaN/Inf values"
        assert torch.isfinite(all_features).all(), "all_features contains NaN/Inf values"    

        # 归一化特征向量
        img_feature = F.normalize(img_feature, p=2, dim=1)
        all_features = F.normalize(all_features, p=2, dim=1)

        # 计算余弦相似度 [-1,1]
        cosine_sim = torch.mm(all_features, img_feature.t()).squeeze(1)

        # 转换到[0,1]范围
        cosine_sim = (cosine_sim + 1) / 2  # 方法1：线性缩放
        # cosine_sim = torch.sigmoid(cosine_sim)  # 方法2：sigmoid

        # 确保数值稳定性
        cosine_sim = torch.clamp(cosine_sim, 0.0, 1.0)

        return cosine_sim


    num_loops = 10

    processed_paths = set()

    all_chosen_rewards = []
    all_chosen_losses = []
    all_chosen_images = []
    all_chosen_eegs = []

    history_cs = []
    history_eeg = []
    history_loss = []
    fit_images = []
    fit_eegs = []
    fit_rewards = []
    fit_losses = []
    save_folder = f'/home/ldy/Workspace/Closed_loop_optimizing/outputs/iclr2026'

    test_set_img_embeds = torch.load("/mnt/dataset1/ldy/Workspace/FLORA/data_preparing/ViT-H-14_features_test.pt")['img_features'].cpu()

    # test_set_img_embeds = torch.load("/home/ldy/Workspace/Closed_loop_optimizing/data/clip_embed/open_clip/600_image_embeds.pt").cpu()
    # 确保基础目录存在
    os.makedirs(save_folder, exist_ok=True)
    
    # 创建基于时间戳的目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_dir = os.path.join(save_folder, timestamp)
    os.makedirs(timestamp_dir, exist_ok=True)
    
    # 在时间戳目录下查找已存在的实验目录
    existing_exps = [d for d in os.listdir(timestamp_dir) if d.startswith('exp') and d[3:].isdigit()]
    if existing_exps:
        # 获取最大的实验编号
        max_num = max(int(exp[3:]) for exp in existing_exps)
        new_exp_num = max_num + 1
    else:
        # 如果没有已存在的实验，从1开始
        new_exp_num = 1
    # 创建新实验目录
    new_exp_dir = os.path.join(timestamp_dir, f'exp{new_exp_num}')
    os.makedirs(new_exp_dir, exist_ok=True)
    print(f"Created new experiment directory: {new_exp_dir}")
    plots_save_folder = f'{new_exp_dir}/plots'
    os.makedirs(plots_save_folder, exist_ok=True)

    Generator = HeuristicGenerator(pipe, vlmodel, preprocess_train, device=device)

    for t in range(num_loops):
        print(f"Loop {t + 1}/{num_loops}")
        loop_save_dir = os.path.join(new_exp_dir, f'loop_{t+1}')
        os.makedirs(loop_save_dir, exist_ok=True)
        loop_sample_ten = []
        loop_reward_ten = []
        loop_eeg_ten = []
        loop_loss_ten = []

        if t == 0:
            chosen_rewards, chosen_losses, chosen_images, chosen_eegs = get_prob_random_sample(test_images_path, 
                                                                                        encoding_model_path, 
                                                                                        fs, device, 
                                                                                        selected_channel_idxes,
                                                                                        processed_paths, 
                                                                                        target_feature,
                                                                                        size=4)

            loop_sample_ten.extend(chosen_images)
            loop_eeg_ten.extend(chosen_eegs)
            loop_reward_ten.extend(chosen_rewards)
            loop_loss_ten.extend(chosen_losses)

            tensor_loop_sample_ten = [preprocess_train(i) for i in loop_sample_ten]    
            with torch.no_grad():
                tensor_loop_sample_ten_embeds = vlmodel.encode_image(torch.stack(tensor_loop_sample_ten).to(device))        
            Generator.pseudo_target_model.add_model_data(torch.tensor(tensor_loop_sample_ten_embeds).to(device), (-torch.tensor(loop_reward_ten) * Generator.reward_scaling_factor).to(device))        

        else:                            
            # loop_sample_ten.extend(fit_images)
            # loop_eeg_ten.extend(fit_eegs)
            # loop_reward_ten.extend(fit_rewards)
            # loop_loss_ten.extend(fit_losses)

            tensor_fit_images = [preprocess_train(i) for i in fit_images]    
            with torch.no_grad():
                img_embeds = vlmodel.encode_image(torch.stack(tensor_fit_images).to(device))    
            generated_images = fusion_image_to_images(Generator, img_embeds, fit_rewards, device, 512, save_path=loop_save_dir)                                    
            synthetic_eegs = generate_eeg_from_image(encoding_model_path, generated_images, device)        

            loop_sample_ten.extend(generated_images)
            loop_eeg_ten.extend(synthetic_eegs)

            for idx, eeg in enumerate(synthetic_eegs):  
                cs, eeg_feature = reward_function_clip_embed(eeg, eeg_model, target_feature, sub, dnn)
                loss = calculate_loss_clip_embed()
                loop_reward_ten.append(cs)
                loop_loss_ten.append(loss)


            greedy_images = []
            # with torch.no_grad():
            #     loop_img_embeds = vlmodel.encode_image(torch.stack([preprocess_train(i) for i in loop_sample_ten]).to(device))

            # 定义要从中随机选择的top K数量
            TOP_K = 10  # 可以根据需要调整这个值

            for img_embed in img_embeds:      
                available_indices = []
                # available_paths = []            
                for i, path in enumerate(test_images_path):
                    if path not in processed_paths:
                        available_indices.append(i)
                        # available_paths.append(path)

                sample_image_paths =[]            
                available_features = test_set_img_embeds[available_indices]

                cosine_similarities = compute_embed_similarity(img_embed.to(device), available_features.to(device))    
                sorted_available_indices = np.argsort(cosine_similarities.cpu())

                # 获取top K的索引（相似度最高的K个）
                top_indices = sorted_available_indices[-TOP_K:]

                # 从top K中随机选择一个
                selected_idx = np.random.choice(top_indices)
                # print(f"available_paths {len(available_paths)}")            
                # print(f"available_indices {available_indices}")
                # print(f"selected_idx {selected_idx}")
                greedy_image = Image.open(test_images_path[selected_idx]).convert("RGB")
                greedy_images.append(greedy_image)
                sample_image_paths.append(test_images_path[selected_idx])

                processed_paths.update(sample_image_paths)   

            synthetic_eegs = generate_eeg_from_image(encoding_model_path, greedy_images, device)            

            loop_sample_ten.extend(greedy_images)
            loop_eeg_ten.extend(synthetic_eegs)

            for idx, eeg in enumerate(synthetic_eegs):  
                cs, eeg_feature = reward_function_clip_embed(eeg, eeg_model, target_feature, sub, dnn)
                loss = calculate_loss_clip_embed()
                loop_reward_ten.append(cs)
                loop_loss_ten.append(loss)


            loop_probabilities = softmax(loop_reward_ten)    
            chosen_rewards, chosen_losses, chosen_images, chosen_eegs = select_from_images(loop_probabilities, loop_reward_ten, loop_loss_ten, loop_sample_ten, loop_eeg_ten, size=4)        

            # 将四个列表按照chosen_rewards的值从大到小排序
            combined = list(zip(chosen_rewards, chosen_losses, chosen_images, chosen_eegs))
            combined.sort(reverse=True, key=lambda x: x[0])  # 按rewards降序排列

            # 解压排序后的数据
            chosen_rewards, chosen_losses, chosen_images, chosen_eegs = zip(*combined)

            # 如果需要将结果转回列表（因为zip返回的是元组）
            chosen_rewards = list(chosen_rewards)
            chosen_losses = list(chosen_losses)
            chosen_images = list(chosen_images)
            chosen_eegs = list(chosen_eegs)


        fit_images = chosen_images
        fit_eegs = chosen_eegs
        fit_rewards = chosen_rewards
        fit_losses = chosen_losses


        all_chosen_rewards.extend(chosen_rewards)
        all_chosen_losses.extend(chosen_losses)
        all_chosen_images.extend(chosen_images)
        all_chosen_eegs.extend(chosen_eegs)

        # print(f"chosen_images {len(chosen_images)}")
        # print(f"chosen_rewards {chosen_rewards}")
        tensor_loop_sample_ten = [preprocess_train(i) for i in loop_sample_ten]    
        with torch.no_grad():
            tensor_loop_sample_ten_embeds = vlmodel.encode_image(torch.stack(tensor_loop_sample_ten).to(device))        
        Generator.pseudo_target_model.add_model_data(torch.tensor(tensor_loop_sample_ten_embeds).to(device), (-torch.tensor(loop_reward_ten) * Generator.reward_scaling_factor).to(device))        
        visualize_top_images(loop_sample_ten, loop_reward_ten, loop_save_dir, t)

        # max_similarity = max(loop_reward_ten)

        # max_index = loop_reward_ten.index(max_similarity)
        # # corresponding_loss = chosen_losses[max_index]


        # if len(history_cs) == 0:
        #     history_cs.append(max_similarity)
        #     # history_loss.append(corresponding_loss) 
        # else:
        #     max_history = max(history_cs)
        #     if max_similarity > max_history:
        #         history_cs.append(max_similarity)
        #         # history_loss.append(corresponding_loss)
        #     else:
        #         history_cs.append(max_history)
        #         # history_loss.append(history_loss[-1])

        # if len(history_cs) >= 2:
        #     if history_cs[-1] != history_cs[-2]:
        #         diff = abs(history_cs[-1] - history_cs[-2])
        #         print(history_cs[-1], history_cs[-2], diff)
        #         if diff <= 1e-4:
        #             print("The difference is within 10e-4, stopping.")
        #             break

        max_similarity = max(loop_reward_ten)
        max_index = loop_reward_ten.index(max_similarity)
        corresponding_eeg = loop_eeg_ten[max_index]  # 获取对应的脑电数据

        if len(history_cs) == 0:
            history_cs.append(max_similarity)
            history_eeg.append(corresponding_eeg)  # 同时记录脑电数据
        else:
            max_history = max(history_cs)
            if max_similarity > max_history:
                history_cs.append(max_similarity)
                history_eeg.append(corresponding_eeg)  # 记录新的脑电数据
            else:
                history_cs.append(max_history)
                history_eeg.append(history_eeg[-1])  # 保持之前的脑电数据

        if len(history_cs) >= 2:
            if history_cs[-1] != history_cs[-2]:
                diff = abs(history_cs[-1] - history_cs[-2])
                print(history_cs[-1], history_cs[-2], diff)
                if diff <= 1e-4:
                    print("The difference is within 10e-4, stopping.")
                    break

    history_cs

    print(f"chosen_rewards {len(chosen_rewards)}")
    print(f"all_chosen_losses {len(all_chosen_losses)}")
    print(f"all_chosen_images {len(all_chosen_images)}")
    print(f"all_chosen_eegs {len(all_chosen_eegs)}")

    plt.figure(figsize=(10, 5))
    plt.plot(history_cs, marker='o', markersize=3, label='Similarity')
    # plt.plot(history_cs, marker='o', markersize=5, label='Similarity')
    # plt.plot(history_loss, marker='x', markersize=3, label='Loss')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend() 
    path = os.path.join(save_path, 'similarities.jpg')
    plt.savefig(path)
    plt.show()

if __name__ == '__main__':
    for i in range(200):
        if i<84: continue
        main_loop(i)
        