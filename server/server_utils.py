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
from PIL import Image as PILImage # Renamed to avoid conflict with wandb.Image
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # Added GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC # Added SVC
from torchvision import models
import einops # Used in the HeuristicGenerator class
from PIL import Image

# Local application/library imports
from model.ATMS_retrieval import get_eeg_features
from model.pseudo_target_model import PseudoTargetModel
from model.utils import generate_eeg


def get_binary_labels(labels):
    """
    Convert labels to binary.
    """
    positive_emotions = ['Amu', 'Ins', 'Ten']
    labels = np.where(np.isin(labels, positive_emotions), 1, 0)
    return labels
    

def get_selected_channel_idxes(data, fs=250, n_selected_channels=3):
    """
    Select channels with the least PSD feature similarity.
    
    :param data: EEG data, shape (n_samples, n_channels, n_timepoints)
    :param fs: Sampling frequency, default 250 Hz
    :return: Indices of the most distinct (least similar) channels
    """
    n_channels = data.shape[1]
    psds = []

    # Compute average PSD for each channel
    for channel_idx in range(n_channels):
        channel_data = data[:, channel_idx, :]  # (n_samples, n_timepoints)
        psd_sum = 0
        for sample_idx in range(channel_data.shape[0]):
            psd, _ = psd_array_multitaper(channel_data[sample_idx], fs, adaptive=True, normalization='full', verbose=0)
            psd_sum += psd
        psds.append(psd_sum / channel_data.shape[0])  # Average PSD for this channel

    psds = np.array(psds)  # Convert to NumPy array, shape (n_channels, n_frequencies)

    # Compute inter-channel similarity
    similarity_matrix = cosine_similarity(psds)
    np.fill_diagonal(similarity_matrix, np.nan)  # Set diagonal (self-similarity) to NaN

    # Compute mean similarity of each channel with all others
    mean_similarity = np.nanmean(similarity_matrix, axis=1)

    # Select channels with the lowest mean similarity
    selected_channel_idxes = np.argsort(mean_similarity)[:n_selected_channels].tolist()
    
    return selected_channel_idxes


def extract_emotion_psd_features(eeg_data, labels, fs=250, selected_channel_idxes=None):
    """
    Extract PSD features from EEG data with corresponding emotion labels.
    
    :param eeg_data: EEG data, shape (n_samples, n_channels, n_timepoints)
    :param labels: Label for each sample (1: positive, 2: negative)
    :param fs: Sampling rate, default 250Hz
    :param selected_channel_idxes: List of channel indices to use. If None, all channels are used
    :return: features (n_samples, n_features), labels (n_samples,)
    """
    features = []
    valid_labels = []
    print(f"========= Extracting features from {len(eeg_data)} samples =========")
    for i in range(len(eeg_data)):
        eeg_sample = eeg_data[i]  # Process single sample: (n_channels, n_timepoints)
        if selected_channel_idxes:
            eeg_sample = eeg_sample[selected_channel_idxes, :]

        # Compute power spectral density
        psd, _ = psd_array_multitaper(eeg_sample, fs, adaptive=True, normalization='full', verbose=0)
        psd_flat = psd.flatten() # Flatten 2D PSD matrix (channels x frequencies) to 1D vector
        print(psd_flat.mean(), psd_flat.std()) # Print mean and standard deviation
        features.append(psd_flat) # Append to feature list
        valid_labels.append(labels[i]) # Append corresponding label

    features = np.array(features)
    valid_labels = np.array(valid_labels)
    return features, valid_labels
 


def train_emotion_classifier(features, labels, test_size=0.2, random_state=42):
    """Train classifier with grid search hyperparameter optimization"""

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, 
                                                       random_state=random_state, stratify=labels)
    
    # Create processing pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Feature standardization
        ('classifier', RandomForestClassifier(random_state=random_state))
    ])
    
    # Parameter grid
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    # Grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'best_emotion_model.pkl')

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    custom_threshold = 0.5
    y_pred_custom = (y_prob >= custom_threshold).astype(int)
    report = classification_report(y_test, y_pred_custom)
    
    print(f"Best parameters: {grid_search.best_params_}")
    
    return best_model, report, y_test, y_pred_custom

def train_svc(features, labels, test_size=0.2, random_state=42):
    """
    Train an SVM classifier.
    
    Args:
    - features: Extracted feature data
    - labels: Corresponding labels
    - test_size: Proportion of test set
    - random_state: Random seed
    
    Returns:
    - clf: Trained classifier
    - report: Classification report
    - y_test: Ground truth labels of the test set
    - y_pred: Predicted labels of the test set
    """
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Feature standardization (important! SVM is very sensitive to feature scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Parameter grid search
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'linear']
    }
    
    # Find best parameters using grid search
    grid_search = GridSearchCV(
        SVC(probability=True, random_state=random_state),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"SVC best parameters: {grid_search.best_params_}")
    
    # Build and train model with best parameters
    clf = grid_search.best_estimator_
    
    # Evaluate model on test set
    y_pred = clf.predict(X_test_scaled)
    report = classification_report(y_test, y_pred)
    
    return clf, report, y_test, y_pred, scaler

def train_gradient_boosting_classifier(features, labels, test_size=0.2, random_state=42):
    """
    Train a Gradient Boosting classifier.
    
    Args:
    - features: Extracted feature data
    - labels: Corresponding labels
    - test_size: Proportion of test set
    - random_state: Random seed
    
    Returns:
    - clf: Trained classifier
    - report: Classification report
    - y_test: Ground truth labels of the test set
    - y_pred: Predicted labels of the test set
    """
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Parameter grid search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
    
    # Find best parameters using grid search
    grid_search = GridSearchCV(
        GradientBoostingClassifier(random_state=random_state),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"GBC best parameters: {grid_search.best_params_}")
    
    # Build and train model with best parameters
    clf = grid_search.best_estimator_
    
    # Evaluate model on test set
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    return clf, report, y_test, y_pred

def baseline_correction(eeg_data, baseline_window=(-250, 0), stimulus_onset=250):
    """
    Perform baseline correction on EEG data.
    
    Args:
    eeg_data: EEG data array or file path
    baseline_window: Baseline window range (in milliseconds, relative to stimulus onset)
    stimulus_onset: Index of stimulus onset in the data
    
    Returns:
    baseline_corrected_data: Baseline-corrected EEG data
    """
    # Load EEG data if a file path is provided
    if isinstance(eeg_data, str):
        eeg_data = np.load(eeg_data)
    
    # At 250Hz sampling rate, 1ms = 0.25 samples; compute sample indices directly
    baseline_start = stimulus_onset + baseline_window[0]
    baseline_end = stimulus_onset + baseline_window[1]
    
    # Ensure indices are within valid range
    baseline_start = max(0, baseline_start)
    baseline_end = min(eeg_data.shape[1], baseline_end)
    
    # Compute mean during the baseline period
    baseline_mean = np.mean(eeg_data[:, baseline_start:baseline_end], axis=1, keepdims=True)
    
    # Subtract baseline mean for correction
    baseline_corrected_data = eeg_data - baseline_mean
    
    return baseline_corrected_data

def create_n_event_npy(data, count=1, fs=250, event_length=5):
    """
    Extract data for the last N events from the recording.
    Adapted for 5-second image presentation + 1-second blank experimental design.
    
    Args:
    data: NumPy array containing EEG data and event markers
    count: Number of last events to return
    
    Returns:
    last_events_data: List of data arrays for the last N events
    """
    # Extract event channel
    event = data[64, :]  # Row 65 stores event markers
    event_indices = np.where(event > 0)[0]  # Find all non-zero event indices
    
    # Adjust count if it exceeds the actual number of events
    count = min(count, len(event_indices))
    
    # Only process the last N events
    last_events = event_indices[-count:] if count > 0 else []
    
    print(f"Found {len(event_indices)} events, processing the last {count}")
    
    # Store processed event data
    event_data_list = []
    
    for idx, event_idx in enumerate(last_events):
        # Design: 5-second image = 1250 samples (at 250Hz sampling rate)
        if event_idx + fs*event_length <= data.shape[1]:  # Ensure index is within bounds (need 5s of stimulus data)
            # Extract data during the event without baseline correction
            event_data = data[:64, event_idx:event_idx + 1250]  # 5 seconds of post-event data
            
            # Append to return list
            event_data_list.append(event_data)
            print(f"Processed event {len(event_indices) - count + idx + 1}, data shape: {event_data.shape}")
        else:
            print(f"Warning: Event {len(event_indices) - count + idx + 1} does not have enough subsequent data")
    
    return event_data_list

def real_time_process(original_data, filters, apply_baseline=False):  # Default: no baseline correction
    """Efficiently process real-time EEG data without baseline correction"""
    # Apply notch filter
    filtered_data = signal.filtfilt(filters['notch'][0], filters['notch'][1], original_data, axis=1)
    
    # Apply bandpass filter
    filtered_data = signal.filtfilt(filters['bandpass'][0], filters['bandpass'][1], filtered_data, axis=1)
    
    # Resample
    if filters['resample_factor'] != 1:
        new_length = int(filtered_data.shape[1] * filters['resample_factor'])
        filtered_data = signal.resample(filtered_data, new_length, axis=1)
        
    return filtered_data

def prepare_filters(fs=250, new_fs=250):
    """Pre-compute all required filters and parameters"""
    # Design notch filter (50Hz power line interference)
    b_notch, a_notch = signal.iirnotch(50, 30, fs)
    
    # Design bandpass filter (1-100Hz)
    b_bp, a_bp = signal.butter(4, [1, 100], btype='bandpass', fs=fs)
    
    # Compute resampling factor
    resample_factor = new_fs/fs
    
    return {
        'notch': (b_notch, a_notch),
        'bandpass': (b_bp, a_bp),
        'resample_factor': resample_factor
    }
    
def compute_embed_similarity(img_feature, all_features):
    """
    Compute cosine similarity between one image and all other images (result in [0, 1]).
    :param img_feature: Feature vector of the selected image [D] or [1, D]
    :param all_features: Feature vectors of all images [N, D]
    :return: Cosine similarities [N] (range 0-1)
    """
    # Ensure inputs are float type
    img_feature = img_feature.float()
    all_features = all_features.float()
    
    # Ensure feature vector is 2D [1, D]
    if img_feature.dim() == 1:
        img_feature = img_feature.unsqueeze(0)
    
    # Check for NaN/Inf values
    assert torch.isfinite(img_feature).all(), "img_feature contains NaN/Inf values"
    assert torch.isfinite(all_features).all(), "all_features contains NaN/Inf values"    
    
    # Normalize feature vectors
    img_feature = F.normalize(img_feature, p=2, dim=1)
    all_features = F.normalize(all_features, p=2, dim=1)
    
    # Compute cosine similarity [-1, 1]
    cosine_sim = torch.mm(all_features, img_feature.t()).squeeze(1)
    
    # Map to [0, 1] range
    cosine_sim = (cosine_sim + 1) / 2  # Method 1: linear scaling
    # cosine_sim = torch.sigmoid(cosine_sim)  # Method 2: sigmoid
    
    # Ensure numerical stability
    cosine_sim = torch.clamp(cosine_sim, 0.0, 1.0)
    
    return cosine_sim


def visualize_top_images(images, similarities, save_folder, iteration):
    """
    Display selected images sorted by similarity using matplotlib.
    :param image_paths: List of image paths
    :param similarities: List of similarity scores for each image
    """
    # Pair images with similarities and sort by similarity in descending order
    image_similarity_pairs = sorted(zip(images, similarities), key=lambda x: x[1], reverse=True)
    
    # Unzip sorted images and similarities
    sorted_images, sorted_similarities = zip(*image_similarity_pairs)

    # Plot images
    fig, axes = plt.subplots(1, len(sorted_images), figsize=(15, 5))
    for i, image in enumerate(sorted_images):
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(f'Similarity: {sorted_similarities[i]:.4f}', fontsize=8)  # Display similarity
    plt.show()
    
    os.makedirs(save_folder, exist_ok=True)  # Create folder if it doesn't exist
    save_path = os.path.join(save_folder, f"visualization_iteration_{iteration}.png")
    fig.savefig(save_path, bbox_inches='tight', dpi=300)  # Save image file
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
    # 1. Check if input data contains invalid values
    if np.isnan(target_signal).any() or np.isinf(target_signal).any():
        print("Warning: Input signal contains NaN or Inf values, replacing with zeros")
        target_signal = np.nan_to_num(target_signal, nan=0.0, posinf=0.0, neginf=0.0)
        
    if np.allclose(target_signal, 0):
        print("=====All zeros!")
    
    # 2. Use more conservative PSD computation parameters
    target_psd, _ = psd_array_multitaper(target_signal, fs, adaptive=True, 
                                        normalization='full', verbose=0)
    
    # 3. Check if PSD result contains invalid values
    if np.isnan(target_psd).any() or np.isinf(target_psd).any():
        print("Warning: PSD computation produced NaN or Inf values, replacing with zeros")
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
    Compute similarity between an image's CLIP embedding and the target feature.
    :param image: Image feature vector [1024]
    :param groundtruth_eeg: Ground truth feature vector [1024]
    :return: Similarity between the image embedding and ground truth
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
    Compute CLIP embedding from EEG data.
    """
    # Ensure EEG data is float32 and add batch dimension
    eeg_tensor = torch.tensor(eeg, dtype=torch.float32).unsqueeze(0)
    
    # Extract features using the EEG model
    eeg_feature = get_eeg_features(eeg_model, eeg_tensor, device, sub)
    
    # Return the EEG feature embedding
    return eeg_feature

def reward_function_clip_embed(eeg, eeg_model, target_feature, sub, device):
    """
    Compute similarity between EEG data and a target feature.
    
    Converts EEG data into a CLIP embedding and computes cosine similarity
    with the target feature. Similarity is normalized to [0, 1] for use as a reward.
    
    Args:
        eeg: NumPy array, raw EEG data with shape (channels, timepoints)
        eeg_model: Pre-trained EEG encoder model for converting EEG data to feature vectors
        target_feature: Target feature vector for similarity computation
        sub: String, subject ID used for feature extraction
        dnn: String, type of deep neural network model
        device: Computation device (cuda or cpu)
        
    Returns:
        similarity: float, cosine similarity normalized to [0, 1]
        eeg_feature: torch.Tensor, extracted EEG feature vector
    """
    # Ensure EEG data is float32 and add batch dimension
    eeg_tensor = torch.tensor(eeg, dtype=torch.float32).unsqueeze(0)
    
    # Extract features using the EEG model
    eeg_feature = get_eeg_features(eeg_model, eeg_tensor, device, sub)
    
    # Compute cosine similarity between EEG feature and target feature
    similarity = torch.nn.functional.cosine_similarity(eeg_feature.to(device), target_feature.to(device))
    
    # Normalize similarity from [-1, 1] to [0, 1]
    similarity = (similarity + 1) / 2
    
    # Return scalar similarity value and EEG feature vector
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
        # Use all channels
        selected_channel_idxes = range(eeg.shape[0])     
    selected_eeg = eeg[selected_channel_idxes, :]
    psd, _ = psd_array_multitaper(selected_eeg, fs, adaptive=True, normalization='full', verbose=0)
    psd = torch.from_numpy(psd.flatten()).unsqueeze(0)
    # print(f"F.cosine_similarity(target_feature, psd) {F.cosine_similarity(target_feature, psd)}")
    # print(f"target_feature {target_feature}")
    return F.cosine_similarity(target_feature, psd).item()

def fusion_image_to_images(Generator, img_embeds, rewards, device, save_path, scale):        
        # Randomly select two different indices
    idx1, idx2 = random.sample(range(len(img_embeds)), 2)
    # Get corresponding embedding vectors and add batch dimension
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
    Convert raw EEG data (64, 1250) to a specific channel ordering (17, 1250) or downsampled (17, 250).
    
    Args:
        eeg_data: Raw EEG data with shape (64, 1250)
        downsample: Whether to downsample, default True, taking every 5th sample
        
    Returns:
        selected_eeg: EEG data with selected channels and downsampling, shape (17, 250) or (17, 1250)
    """
    # Ensure input data has correct shape
    if eeg_data.shape[0] != 64:
        raise ValueError(f"Input EEG data should have 64 channels, but got {eeg_data.shape[0]}")
    
    # Define required channel indices (0-based)
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
        53,  # PO5 (substitute for P1)
        43,  # Pz
        54,  # PO6 (substitute for P2)
        45,  # P4
        47,  # P6
        49   # P8
    ]
    
    # Extract required channels
    selected_eeg = eeg_data[channel_indices, :]
    
    # Downsample by taking every 5th sample
    # if downsample:
    #     selected_eeg = selected_eeg[:, ::5]  # Slice to take every 5th sample
    
    # Take the first 250 data points
    selected_eeg = selected_eeg[:, :250]
    
    print(f"Original EEG data shape: {eeg_data.shape}")
    print(f"Converted EEG data shape: {selected_eeg.shape}")
    
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