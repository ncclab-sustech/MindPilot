import os
import random
from scipy import signal

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from mne.time_frequency import psd_array_multitaper
from PIL import Image as PILImage # 重命名以避免与 wandb.Image 冲突
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # 添加了 GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC # 添加了 SVC
from torchvision import models
import einops # 在 HeuristicGenerator 类中使用
from PIL import Image

# 本地应用/库导入
from model.ATMS_retrieval import get_eeg_features
from model.pseudo_target_model import PseudoTargetModel
from model.utils import generate_eeg


def get_binary_labels(labels):
    """
    把标签转换为二进制。
    """
    positive_emotions = ['Amu', 'Ins', 'Ten']
    labels = np.where(np.isin(labels, positive_emotions), 1, 0)
    return labels
    

def get_selected_channel_idxes(data, fs=250, n_selected_channels=3):
    """
    挑选出 PSD 特征之间相似度最小的几个通道。
    
    :param data: EEG 数据，形状为 (n_samples, n_channels, n_timepoints)
    :param fs: 采样频率，默认 250 Hz
    :return: 最显著（相似度最小）的三个通道索引
    """
    n_channels = data.shape[1]
    psds = []

    # 计算每个通道的平均 PSD
    for channel_idx in range(n_channels):
        channel_data = data[:, channel_idx, :]  # (n_samples, n_timepoints)
        psd_sum = 0
        for sample_idx in range(channel_data.shape[0]):
            psd, _ = psd_array_multitaper(channel_data[sample_idx], fs, adaptive=True, normalization='full', verbose=0)
            psd_sum += psd
        psds.append(psd_sum / channel_data.shape[0])  # 计算通道的平均 PSD

    psds = np.array(psds)  # 转为 NumPy 数组，形状为 (n_channels, n_frequencies)

    # 计算通道之间的相似度
    similarity_matrix = cosine_similarity(psds)
    np.fill_diagonal(similarity_matrix, np.nan)  # 将对角线（自相似）设置为 NaN

    # 计算每个通道与其他通道的平均相似度
    mean_similarity = np.nanmean(similarity_matrix, axis=1)

    # 选取平均相似度最小的几个通道
    selected_channel_idxes = np.argsort(mean_similarity)[:n_selected_channels].tolist()
    
    return selected_channel_idxes


def extract_emotion_psd_features(eeg_data, labels, fs=250, selected_channel_idxes=None):
    """
    提取 EEG 数据的 PSD 特征并对应情绪标签。
    
    :param eeg_data: EEG 数据，形状为 (n_samples, n_channels, n_timepoints)
    :param labels: 每个样本对应的标签（1: positive, 2: negative）
    :param fs: 采样率，默认 250Hz
    :param selected_channel_idxes: 指定的通道索引列表。如果为 None，则使用所有通道
    :return: features (n_samples, n_features), la   axbels (n_samples,)
    """
    features = []
    valid_labels = []
    print(f"========= Extracting features from {len(eeg_data)} samples =========")
    for i in range(len(eeg_data)):
        eeg_sample = eeg_data[i]  # 处理单个样本：(n_channels, n_timepoints)
        if selected_channel_idxes:
            eeg_sample = eeg_sample[selected_channel_idxes, :]

        # 计算功率谱密度
        psd, _ = psd_array_multitaper(eeg_sample, fs, adaptive=True, normalization='full', verbose=0)
        psd_flat = psd.flatten() # 将2D的PSD矩阵(通道×频率)展平为1D向量
        print(psd_flat.mean(), psd_flat.std()) # 打印均值和标准差
        features.append(psd_flat) # 添加到特征列表
        valid_labels.append(labels[i]) # 添加对应的标签

    features = np.array(features)
    valid_labels = np.array(valid_labels)
    return features, valid_labels
 


def train_emotion_classifier(features, labels, test_size=0.2, random_state=42):
    """分类器训练，使用网格搜索优化参数"""

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, 
                                                       random_state=random_state, stratify=labels)
    
    # 创建处理管道
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # 特征标准化
        ('classifier', RandomForestClassifier(random_state=random_state))
    ])
    
    # 参数网格
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    # 网格搜索
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # 最佳模型
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'best_emotion_model.pkl')

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    custom_threshold = 0.5
    y_pred_custom = (y_prob >= custom_threshold).astype(int)
    report = classification_report(y_test, y_pred_custom)
    
    print(f"最佳参数: {grid_search.best_params_}")
    
    return best_model, report, y_test, y_pred_custom

def train_svc(features, labels, test_size=0.2, random_state=42):
    """
    训练SVM分类器
    
    参数:
    - features: 提取的特征数据
    - labels: 对应的标签
    - test_size: 测试集比例
    - random_state: 随机数种子
    
    返回:
    - clf: 训练好的分类器
    - report: 分类报告
    - y_test: 测试集的真实标签
    - y_pred: 测试集的预测标签
    """
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # 特征标准化（重要！SVM对特征缩放非常敏感）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 参数网格搜索
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'linear']
    }
    
    # 使用网格搜索找到最佳参数
    grid_search = GridSearchCV(
        SVC(probability=True, random_state=random_state),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"SVC最佳参数: {grid_search.best_params_}")
    
    # 使用最佳参数构建和训练模型
    clf = grid_search.best_estimator_
    
    # 在测试集上评估模型
    y_pred = clf.predict(X_test_scaled)
    report = classification_report(y_test, y_pred)
    
    return clf, report, y_test, y_pred, scaler

def train_gradient_boosting_classifier(features, labels, test_size=0.2, random_state=42):
    """
    训练梯度提升分类器
    
    参数:
    - features: 提取的特征数据
    - labels: 对应的标签
    - test_size: 测试集比例
    - random_state: 随机数种子
    
    返回:
    - clf: 训练好的分类器
    - report: 分类报告
    - y_test: 测试集的真实标签
    - y_pred: 测试集的预测标签
    """
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.ensemble import GradientBoostingClassifier
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # 参数网格搜索
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
    
    # 使用网格搜索找到最佳参数
    grid_search = GridSearchCV(
        GradientBoostingClassifier(random_state=random_state),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"GBC最佳参数: {grid_search.best_params_}")
    
    # 使用最佳参数构建和训练模型
    clf = grid_search.best_estimator_
    
    # 在测试集上评估模型
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    return clf, report, y_test, y_pred

def baseline_correction(eeg_data, baseline_window=(-250, 0), stimulus_onset=250):
    """
    对EEG数据进行基线校正
    
    参数:
    eeg_data: EEG数据数组或文件路径
    baseline_window: 基线窗口范围(以毫秒为单位，相对于刺激呈现时间点)
    stimulus_onset: 在数据中刺激呈现的时间点索引
    
    返回:
    baseline_corrected_data: 基线校正后的EEG数据
    """
    # 加载EEG数据，如果传入的是路径
    if isinstance(eeg_data, str):
        eeg_data = np.load(eeg_data)
    
    # 采样率为250Hz时，1ms = 0.25点，直接计算点数
    baseline_start = stimulus_onset + baseline_window[0]
    baseline_end = stimulus_onset + baseline_window[1]
    
    # 确保索引在有效范围内
    baseline_start = max(0, baseline_start)
    baseline_end = min(eeg_data.shape[1], baseline_end)
    
    # 计算基线期间的平均值
    baseline_mean = np.mean(eeg_data[:, baseline_start:baseline_end], axis=1, keepdims=True)
    
    # 减去基线均值进行校正
    baseline_corrected_data = eeg_data - baseline_mean
    
    return baseline_corrected_data

def create_n_event_npy(data, count=1, fs=250, event_length=5):
    """
    从数据中提取最后count个事件的数据
    适配5秒图片呈现+1秒空白的实验设计
    
    参数:
    data: 包含EEG数据和事件标记的numpy数组
    count: 返回最后几个事件的数据
    
    返回:
    last_events_data: 最后count个事件的数据数组列表
    """
    # 提取事件通道
    event = data[64, :]  # 第65行存储event信息
    event_indices = np.where(event > 0)[0]  # 找到所有非零的event索引
    
    # 如果要求的事件数大于实际事件数，调整count
    count = min(count, len(event_indices))
    
    # 只处理最后count个事件
    last_events = event_indices[-count:] if count > 0 else []
    
    print(f"找到 {len(event_indices)} 个事件，处理最后 {count} 个")
    
    # 储存处理后的事件数据
    event_data_list = []
    
    for idx, event_idx in enumerate(last_events):
        # 新的设计: 5秒图片 = 1250个样本点 (采样率250Hz)
        if event_idx + fs*event_length <= data.shape[1]:  # 确保索引不越界（需要5秒的刺激数据）
            # 直接提取事件期间的数据，不进行基线校正
            event_data = data[:64, event_idx:event_idx + 1250]  # 事件后5秒数据
            
            # 添加到返回列表
            event_data_list.append(event_data)
            print(f"处理了事件 {len(event_indices) - count + idx + 1}，数据形状: {event_data.shape}")
        else:
            print(f"警告: 事件 {len(event_indices) - count + idx + 1} 没有足够的后续数据")
    
    return event_data_list

def real_time_process(original_data, filters, apply_baseline=False):  # 默认不应用基线校正
    """高效处理实时EEG数据，不包括基线校正"""
    # 应用陷波滤波器
    filtered_data = signal.filtfilt(filters['notch'][0], filters['notch'][1], original_data, axis=1)
    
    # 应用带通滤波器
    filtered_data = signal.filtfilt(filters['bandpass'][0], filters['bandpass'][1], filtered_data, axis=1)
    
    # 重采样
    if filters['resample_factor'] != 1:
        new_length = int(filtered_data.shape[1] * filters['resample_factor'])
        filtered_data = signal.resample(filtered_data, new_length, axis=1)
        
    return filtered_data

def prepare_filters(fs=250, new_fs=250):
    """预先计算所有需要的滤波器和参数"""
    # 设计陷波滤波器（50Hz电源干扰）
    b_notch, a_notch = signal.iirnotch(50, 30, fs)
    
    # 设计带通滤波器（1-100Hz）
    b_bp, a_bp = signal.butter(4, [1, 100], btype='bandpass', fs=fs)
    
    # 计算重采样因子
    resample_factor = new_fs/fs
    
    return {
        'notch': (b_notch, a_notch),
        'bandpass': (b_bp, a_bp),
        'resample_factor': resample_factor
    }
    
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
    
# def load_target_feature(target_path, fs, selected_channel_idxes):
#     target_signal = np.load(target_path, allow_pickle=True)
#     print(f"target_signal shape: {target_signal.shape}")
#     # noise = torch.randn(size=(3, 250))
#     target_psd, _ = psd_array_multitaper(target_signal, fs, adaptive=True, normalization='full', verbose=0)
#     print(f"Target psd shape:{target_psd.shape}")
#     return torch.from_numpy(target_psd.flatten()).unsqueeze(0)

def load_target_feature(target_path, fs, selected_channel_idxes):
    target_signal = np.load(target_path, allow_pickle=True)
    target_signal = target_signal[selected_channel_idxes, :]
    print(f"target_signal shape: {target_signal.shape}")
    print(f"{target_signal}")
    # 1. 检查输入数据是否包含无效值
    if np.isnan(target_signal).any() or np.isinf(target_signal).any():
        print("警告: 输入信号包含 NaN 或 Inf 值，将替换为零")
        target_signal = np.nan_to_num(target_signal, nan=0.0, posinf=0.0, neginf=0.0)
        
    if np.allclose(target_signal, 0):
        print("=====全零!")
    
    # 2. 使用更保守的 PSD 计算参数
    target_psd, _ = psd_array_multitaper(target_signal, fs, adaptive=True, 
                                        normalization='full', verbose=0)
    
    # 3. 检查 PSD 结果是否包含无效值
    if np.isnan(target_psd).any() or np.isinf(target_psd).any():
        print("警告: PSD 计算产生了 NaN 或 Inf 值，将替换为零")
        target_psd = np.nan_to_num(target_psd, nan=0.0, posinf=0.0, neginf=0.0)
        
    print(f"Target psd shape:{target_psd.shape}")
    return torch.from_numpy(target_psd.flatten()).unsqueeze(0)

def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def generate_eeg_from_image_paths(model_path, test_image_list, save_dir, device):
    synthetic_eegs = []
    model = load_model_encoder(model_path, device)
    for idx, image_path in enumerate(test_image_list):
        image_tensor = preprocess_image(image_path, device)
        synthetic_eeg = generate_eeg(model, image_tensor, device)
        synthetic_eegs.append(synthetic_eeg)

    return synthetic_eegs

def load_model_encoder(model_path, device):
    model = create_model(device, 'alexnet')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['best_model'])
    model.eval()
    return model  

def create_model(device, dnn):
    if dnn == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = torch.nn.Linear(4096, 4250)
    model = model.to(device)
    return model 
     
def generate_eeg_from_image(model_path, images, save_dir, device):
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

def reward_function_clip_embed_image(pil_image, target_feature, device, vlmodel, preprocess_train):
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

def get_target_feature_from_eeg(eeg, eeg_model, device, sub):
    """
    计算EEG clip embed
    """
    # 确保EEG数据是float32类型，并添加批次维度
    eeg_tensor = torch.tensor(eeg, dtype=torch.float32).unsqueeze(0)
    
    # 使用EEG模型提取特征
    eeg_feature = get_eeg_features(eeg_model, eeg_tensor, device, sub)
    
    # 计算EEG特征与目标特征之间的余弦相似度
    return eeg_feature

def reward_function_clip_embed(eeg, eeg_model, target_feature, sub, device):
    """
    计算EEG数据与目标特征的相似度。
    
    该函数将EEG数据转换为CLIP嵌入特征，并与目标特征计算余弦相似度。
    相似度被归一化到[0,1]范围，以便用作奖励值。
    
    参数:
        eeg: numpy数组，原始EEG数据，形状为(channels, timepoints)
        eeg_model: 预训练的EEG编码模型，用于将EEG数据转换为特征向量
        target_feature: 目标特征向量，用于与EEG特征计算相似度
        sub: 字符串，被试ID，用于特征提取
        dnn: 字符串，使用的深度神经网络模型类型
        device: 计算设备(cuda或cpu)
        
    返回:
        similarity: float, 归一化到[0,1]范围的余弦相似度
        eeg_feature: torch.Tensor, 提取的EEG特征向量
    """
    # 确保EEG数据是float32类型，并添加批次维度
    eeg_tensor = torch.tensor(eeg, dtype=torch.float32).unsqueeze(0)
    
    # 使用EEG模型提取特征
    eeg_feature = get_eeg_features(eeg_model, eeg_tensor, device, sub)
    
    # 计算EEG特征与目标特征之间的余弦相似度
    similarity = torch.nn.functional.cosine_similarity(eeg_feature.to(device), target_feature.to(device))
    
    # 将相似度从[-1,1]范围归一化到[0,1]范围
    similarity = (similarity + 1) / 2
    
    # 返回标量相似度值和EEG特征向量
    return similarity.item(), eeg_feature

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
    if selected_channel_idxes is None or len(selected_channel_idxes) == 0:
        # 使用所有通道
        selected_channel_idxes = range(eeg.shape[0])     
    selected_eeg = eeg[selected_channel_idxes, :]
    psd, _ = psd_array_multitaper(selected_eeg, fs, adaptive=True, normalization='full', verbose=0)
    psd = torch.from_numpy(psd.flatten()).unsqueeze(0)
    # print(f"F.cosine_similarity(target_feature, psd) {F.cosine_similarity(target_feature, psd)}")
    # print(f"target_feature {target_feature}")
    return F.cosine_similarity(target_feature, psd).item()

def fusion_image_to_images(Generator, img_embeds, rewards, device, save_path, scale):        
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
    # with torch.no_grad():         
    images = Generator.generate(img_embeds.to(device), torch.tensor(rewards).to(device), prompt='', save_path=None, start_embedding=embed1)
    # image = generator.generate(embed1)
    generated_images.extend(images)
    # print(f"type(images) {type(images)}")
    images = Generator.generate(img_embeds.to(device), torch.tensor(rewards).to(device), prompt='', save_path=None, start_embedding=embed2)
    # image = generator.generate(embed2)
    generated_images.extend(images)
    
    return generated_images
    
def select_from_image_paths(probabilities, similarities, sample_image_paths, synthetic_eegs, size):
    chosen_indices = np.random.choice(len(probabilities), size=size, replace=False, p=probabilities)
    # print(f"sample_image_paths {len(sample_image_paths)}")
    # print(f"chosen_indices  {chosen_indices}")
    
    chosen_similarities = [similarities[idx] for idx in chosen_indices.tolist()] 
    chosen_images = [Image.open(sample_image_paths[i]).convert("RGB") for i in chosen_indices.tolist()]        
    chosen_eegs = [synthetic_eegs[idx] for idx in chosen_indices.tolist()]
    return chosen_similarities, chosen_images, chosen_eegs

def select_from_image_paths_without_eeg(probabilities, similarities, sample_image_paths, size):
    chosen_indices = np.random.choice(len(probabilities), size=size, replace=False, p=probabilities)
    # print(f"sample_image_paths {len(sample_image_paths)}")
    # print(f"chosen_indices  {chosen_indices}")
    
    chosen_similarities = [similarities[idx] for idx in chosen_indices.tolist()]     
    chosen_images = [Image.open(sample_image_paths[i]).convert("RGB") for i in chosen_indices.tolist()]        
    return chosen_similarities, chosen_images

def select_from_images(probabilities, similarities, images_list, eeg_list, size):
    chosen_indices = np.random.choice(len(similarities), size=size, replace=False, p=probabilities)
    # print(f"eeg_list {len(eeg_list)}")
    # print(f"chosen_indices  {chosen_indices}")    
    chosen_similarities = [similarities[idx] for idx in chosen_indices.tolist()] 
    chosen_images = [images_list[idx] for idx in chosen_indices.tolist()]
    chosen_eegs = [eeg_list[idx] for idx in chosen_indices.tolist()]
    return chosen_similarities, chosen_images, chosen_eegs

def select_from_images_without_eeg(probabilities, similarities, images_list, size):
    chosen_indices = np.random.choice(len(similarities), size=size, replace=False, p=probabilities)
    # print(f"eeg_list {len(eeg_list)}")
    # print(f"chosen_indices  {chosen_indices}")    
    chosen_similarities = [similarities[idx] for idx in chosen_indices.tolist()] 
    chosen_images = [images_list[idx] for idx in chosen_indices.tolist()]
    return chosen_similarities, chosen_images

def convert_eeg(eeg_data, downsample=True):
    """
    将原始EEG数据(64,1250)转换为特定通道排序的(17,1250)或降采样后的(17,250)
    
    参数:
        eeg_data: 原始EEG数据，形状为(64,1250)
        downsample: 是否进行降采样，默认为True，每5个点取1个点
        
    返回:
        selected_eeg: 选定通道和降采样后的EEG数据，形状为(17,250)或(17,1250)
    """
    # 确保输入数据形状正确
    if eeg_data.shape[0] != 64:
        raise ValueError(f"输入EEG数据应有64个通道，但实际有{eeg_data.shape[0]}个")
    
    # 定义所需通道的索引（0-based索引）
    channel_indices = [
        58,  # O1
        57,  # Oz
        59,  # O2
        55,  # PO7
        51,  # PO3
        50,  # POz
        52,  # PO4
        56,  # PO8
        48,  # P7
        46,  # P5
        44,  # P3
        53,  # PO5 (替代P1)
        43,  # Pz
        54,  # PO6 (替代P2)
        45,  # P4
        47,  # P6
        49   # P8
    ]
    
    # 提取所需通道
    selected_eeg = eeg_data[channel_indices, :]
    
    # 进行降采样，每5个点取1个
    # if downsample:
    #     selected_eeg = selected_eeg[:, ::5]  # 使用切片操作每5个点取1个
    
    # 取最前面的 250 个数据点
    selected_eeg = selected_eeg[:, :250]
    
    print(f"原始EEG数据形状: {eeg_data.shape}")
    print(f"转换后EEG数据形状: {selected_eeg.shape}")
    
    return selected_eeg

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
        
    def generate(self, data_x, data_y, prompt='', save_path=None, start_embedding=None):
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
            
            data_x, data_y = self.pseudo_target_model.get_model_data()   
            # print(f"data_y.size(0) {data_y.size(0)}")
            if data_y.size(0) < 50: #w/o optimization 
                return images
            
            image_inputs = torch.stack([self.preprocess_train(img) for img in images])
            
            # Get image features and calculate similarity
            with torch.no_grad():
                image_features = self.vlmodel.encode_image(image_inputs.to(self.device)) 
            
            # Use the class method instead of external function
            # scaled_similarity = self.reward_function_embed(
            #     image_features, 
            #     tar_image_embed.expand(self.generate_batch_size, tar_image_embed.size(-1))
            # ) 
            
            # Update pseudo target
            step_size = self.initial_step_size / (1 + self.decay_rate * step)
            # print(f"image_features {image_features.shape}")
            # print(f"image_features {len(image_features)}")

            pseudo_target = self.pseudo_target_model.estimate_pseudo_target(image_features, step_size=step_size) #batchsize, hidden_dim
            
            # Save images periodically
            if step % self.save_per == 0:
                # print(f"scaled_similarity {scaled_similarity}")
                all_images.append(images)
            
            del latents
        # Save merged image if path provided
        if save_path:
            merged_image = self.merge_images_grid(all_images)
            merged_image.save(save_path)
        
        return images