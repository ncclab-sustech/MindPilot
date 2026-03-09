"""
Closed-loop experiment server for ML-based image-EEG-rating collection.
This server implements an interactive experiment framework for collecting user ratings
on images and EEG data, and generating new images through optimization algorithms.

Experimenter Instructions

First, collect two rounds per subject, naming output_save_path as {name}_{male/female}_1 and {name}_{male/female}_2


1. First task:

Set use_eeg to False

Set output_save_path to rating

2. Second task:

Set use_eeg to True

Two feature types: clip and psd, set output_save_path to clip and psd respectively

"""



import base64
import json
import os
import random
import time
import shutil
from threading import Event
import io


import matplotlib.pyplot as plt
import numpy as np
import open_clip  # CLIP model for image feature extraction
from PIL import Image
from scipy.special import softmax
import torch

from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit

# Local application/library imports
from model.custom_pipeline_low_level import Generator4Embeds  # Image generation model
from model.ATMS_retrieval import ATMS  # EEG feature extraction model
from modulation_utils import *  # Wildcard import, includes various image processing tools
from server_utils import *  # Wildcard import, includes server utility functions


proxy = 'http://10.32.204.163:7897'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy
# Set Hugging Face mirror address
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "5" 

# Initialize Flask app and Socket.IO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow cross-origin requests

#====================== Global Experiment Parameters ======================
# Basic experiment settings
feature_type = 'clip'      # Feature type: 'clip', 'psd', 'clip_img'
# output_save_path = '/home/ldy/Closed_loop_optimizing/server/outputs/shenyuyang_male_2/rating'  # Experiment results save path
output_save_path = f'/home/ldy/Closed_loop_optimizing/server/outputs/shenyuyang_male_2/{feature_type}'  # Experiment results save path
sub = 'sub-01'                # Subject ID
subject_id = 1                # Numeric subject ID
fs = 250                      # EEG sampling frequency (Hz)
num_loops = 10                # Number of experiment loops
use_eeg = True            # Whether to use EEG data
device = "cuda" if torch.cuda.is_available() else "cpu"  # Compute device
model_type = 'ViT-H-14'        # CLIP model type
dnn = 'alexnet'                # DNN model type
random.seed(30)           # Random seed
is_post = False         # Whether this is a follow-up to experiment 2

# Data collection containers
processed_paths = set()       # Set of already processed image paths

# Experiment record lists
all_chosen_rewards = []       # All selected reward values
all_chosen_images = []        # All selected images
all_chosen_eegs = []          # All selected EEG data

# History and fitting data
history_cs = []               # Historical similarity records
fit_images = []               # Images for fitting
fit_eegs = []                 # EEG data for fitting
fit_rewards = []              # Reward values for fitting
fit_losses = []               # Loss values for fitting


# Pre-loaded test set embeddings
test_set_img_embeds = torch.load("/mnt/dataset1/ldy/Workspace/FLORA/data_preparing/ViT-H-14_features_test.pt")['img_features'].cpu()

#====================== Path Parameters ======================
# Image and data paths
# image_set_path = 'image_pool_square'  # Image set path
image_set_path = 'test_images'

instant_eeg_path = 'server/data/instant_eeg'                                           # Real-time EEG data storage path
cache_path = 'server/data/cache'           

# target_image_path = 'image_pool_square/square_Dis-07.jpg' 
# target_image_path = 'test_images/00014_bike_bike_22s.jpg'
target_image_path = 'test_images/00131_pear_pear_01b.jpg'

    
target_eeg_path = ''                                # Target EEG data path

#====================== Global Variables ======================
target_eeg_path = None         # Target EEG path
target_feature = None           # Target feature
clf = None                     # Classifier
rating_received_event = Event() # Rating received event (for thread synchronization)
eeg_received_event = Event()   # EEG data received event (for thread synchronization)

# Temporarily stored data
ratings = []                   # User ratings
eeg = None                     # EEG data

#====================== Model Preparation ======================
# Initialize CLIP model and its preprocessing functions
vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
    model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device=device)
vlmodel.to(device)

# Initialize image generator
generator = Generator4Embeds(device=device)
pipe = generator.pipe

if use_eeg:
    # Load different target features and paths based on feature type
    if feature_type == 'psd':
        selected_channel_idxes = range(59)  # Selected EEG channel indices
        
    elif feature_type == 'clip':
        # Load CLIP-encoded EEG embeddings
        # target_eeg_embed = "/mnt/dataset0/xkp/closed-loop/server/target_embed/open_clip/00014_bike_eeg_embeds.pt"
        # Load EEG encoder model
        f_encoder = f"/mnt/dataset0/kyw/closed-loop/sub_model/{sub}/diffusion_alexnet/pretrained_True/gene_gene/ATM_S_reconstruction_scale_0_1000_40.pth"
        checkpoint = torch.load(f_encoder, map_location=device)

        eeg_model = ATMS()  # EEG feature extraction model
        eeg_model.load_state_dict(checkpoint['eeg_model_state_dict'])

    elif feature_type == 'clip_img': 
        # Load CLIP-based image embeddings
        gt_eeg_folder = f'/mnt/dataset0/kyw/closed-loop/syn_eeg_gt'
        # target_image_embed = "/home/ldy/Closed_loop_optimizing/data/clip_embed/open_clip/00135_pie_image_embeds.pt"
        # target_image_path = "/mnt/dataset0/ldy/4090_Workspace/4090_THINGS/images_set/test_images/00135_pie/pie_18s.jpg"

#====================== SocketIO Event Handling ======================
@socketio.on('connect')
def handle_connect(auth=None):
    """Handle client connection event"""
    print('Client connected')
    # print('Send: experiment_2_ready')
    # socketio.emit('experiment_2_ready')  # Notify client that experiment is ready
    socketio.emit('experiment_1_ready')  # Notify client that experiment is ready


#====================== Flask Routes ======================
@app.route('/experiment_1', methods=['POST'])
def experiment_1():
    """Experiment 1: Determine the target"""
    global selected_channel_idxes
    global target_eeg_path 
    global target_image_path
    global target_feature
    global ratings
    
    print("\n" + "#" * 50)
    print("EEG Feature Selection Experiment")
    print("#" * 50 + "\n")
    
    # Create save directory for current experiment
    exp_1_save_path = os.path.join(output_save_path, f'experiment_1')
    os.makedirs(exp_1_save_path, exist_ok=True)
    
    
    if (use_eeg):
        target_eeg_list = send_images_and_collect_ratings_and_eeg([target_image_path], exp_1_save_path, 1)
        target_eeg_path = os.path.join(exp_1_save_path, 'eeg_0.npy')
        eeg = np.load(target_eeg_path, allow_pickle=True)
        print(f"eeg shape: {eeg.shape}")
        # Check target EEG data shape and correctness
        # Load target feature based on feature type
        if feature_type == 'psd':
            target_feature = load_target_feature(target_eeg_path, fs, selected_channel_idxes)
            print(f"target_feature shape: {target_feature.shape}")
        elif feature_type == 'clip':
            target_feature = get_target_feature_from_eeg(eeg, eeg_model, device, sub)
            print(f'target_feature shape: {target_feature.shape}')
    else:
        # Send target image to client and collect ratings
        success = send_images_and_collect_ratings([target_image_path], exp_1_save_path)
        if not success:
            print("Failed to collect ratings, experiment terminated")
            return jsonify({"message": "Failed to collect ratings, experiment terminated"}), 500
        
    print("Experiment 1 finished")
    experiment_2()
    # experiment_2_post()
    # Notify client that experiment is complete
    socketio.emit('experiment_finished', {
        "message": "Experiment completed"
    })
    
    return jsonify({
        "message": "Experiment completed successfully",
    }), 200
    

@app.route('/experiment_2', methods=['POST'])
def experiment_2():
    """Main experiment processing function that generates and optimizes images through iteration"""
    global selected_channel_idxes
    global target_eeg_path
    global target_image_path
    global ratings
    global target_feature

    print("\n" + "#" * 50)
    print("Image Rating Iterative Experiment")
    print("#" * 50 + "\n")
    
    # Initialize heuristic generator
    Generator = HeuristicGenerator(pipe, vlmodel, preprocess_train, device=device)
    
    # Get all images
    test_images = [f for f in os.listdir(image_set_path) if f.endswith('.jpg') or f.endswith('.png')]
    print(f"Total {len(test_images)} images")
    
    # Build image path list
    test_images_path = [os.path.join(image_set_path, test_image) for test_image in test_images]
    
    # Remove target image from test set if present
    if target_image_path in test_images_path:
        test_images_path.remove(target_image_path)
    
    processed_paths = set()  # Track processed image paths
    
    # Experiment data records    
    all_viewed_image_paths = []  # All viewed image paths
    all_greedy_image_paths = []  # All greedy-selected image paths
    all_fusion_image_paths = []  # All fusion-generated image paths
    all_viewed_image_rewards = []  # Rewards for all viewed images
    all_fusion_image_rewards = []  # Rewards for all fusion-generated images
    all_greedy_image_rewards = []  # Rewards for all greedy-selected images
    all_viewed_image_ratings = []  # Ratings for all viewed images (not necessarily rewards)
    all_fusion_image_ratings = []  # Ratings for all fusion-generated images (not necessarily rewards)
    all_greedy_image_ratings = []  # Ratings for all greedy-selected images (not necessarily rewards)

    
    #====================== Main Experiment Loop ======================
    for t in range(num_loops):
        print(f"Loop {t + 1}/{num_loops}")
        
        # Create save directory for current loop
        round_save_path = os.path.join(output_save_path, f'loop{t + 1}')
        os.makedirs(round_save_path, exist_ok=True)
        
        # Data collection containers for current loop
        loop_sample_ten = []     # Sample images for current loop
        loop_reward_ten = []     # Reward values for current loop
        if(use_eeg):
            loop_eeg_ten = []        # EEG data for current loop
        

        #====================== First Round: Random Sampling ======================
        if t == 0:
            # Create save directory for first round
            first_ten = os.path.join(round_save_path, 'first_ten')
            os.makedirs(first_ten, exist_ok=True)
    
            # Randomly select 10 images from unprocessed ones
            available_paths = [path for path in test_images_path if path not in processed_paths]    
            sample_image_paths = sorted(random.sample(available_paths, 10))
            
            # Or, manually select
            # sample_image_paths = [...]
            
            all_viewed_image_paths.extend(sample_image_paths)  # Update viewed image paths
            
            # Load images and prepare for processing
            pil_images = []
            for sample_image_path in sample_image_paths:
                pil_images.append(Image.open(sample_image_path).convert("RGB"))
            
            similarities = []
            eegs=[]
            
            if (use_eeg):
                # Send images to client and collect ratings and EEG data
                eegs = send_images_and_collect_ratings_and_eeg(sample_image_paths, first_ten, 10)
                print(f"eegs length: {len(eegs)}")
                # Compute similarities and losses
                for idx, eeg in enumerate(eegs):  
                    # Compute similarity and loss based on feature type
                    if feature_type == 'psd':
                        cs = reward_function(eeg, target_feature, fs, selected_channel_idxes)
                        
                    elif feature_type == 'clip':
                        cs, eeg_feature = reward_function_clip_embed(eeg, eeg_model, target_feature, sub, device)
                    
                    # Record similarity
                    all_viewed_image_rewards.append(cs) 
                    similarities.append(cs)        
                    
                    # Record rating
                    all_viewed_image_ratings.append(ratings[idx])
                  
                print(f"Similarities: {similarities}")
            else:
                # Send images to client and collect ratings
                success = send_images_and_collect_ratings(sample_image_paths, first_ten)
                if not success:
                    print("Failed to collect ratings, experiment terminated")
                    return jsonify({"message": "Failed to collect ratings, experiment terminated"}), 500
                # Use user ratings as similarity (reward)
                for rating in ratings:
                    similarities.append(rating)
                    all_viewed_image_rewards.append(rating)
                    all_viewed_image_ratings.append(rating)
                # Clear ratings list
                ratings = []
                
            # Compute selection probabilities
            probabilities = softmax(similarities)
            # Update pseudo target model
                        
            # Add data to pseudo target model
            tensor_loop_sample_ten = [preprocess_train(i) for i in pil_images]    
            with torch.no_grad():
                tensor_loop_sample_ten_embeds = vlmodel.encode_image(torch.stack(tensor_loop_sample_ten).to(device))        
            
                        
            Generator.pseudo_target_model.add_model_data(
                torch.tensor(tensor_loop_sample_ten_embeds).to(device), 
                (-torch.tensor(similarities) * Generator.reward_scaling_factor).to(device)
            )
            if (use_eeg):
                # Select images based on probabilities
                chosen_rewards, chosen_images, chosen_eegs = select_from_image_paths(
                    probabilities, similarities, sample_image_paths, eegs, size=4
                )
                # Update current loop data
                loop_sample_ten.extend(chosen_images)
                loop_eeg_ten.extend(chosen_eegs)
                loop_reward_ten.extend(chosen_rewards)
            else:
                chosen_rewards, chosen_images = select_from_image_paths_without_eeg(
                    probabilities, similarities, sample_image_paths, size=4
                )
                # Update current loop data
                loop_sample_ten.extend(chosen_images)
                loop_reward_ten.extend(chosen_rewards)
               
        
        #====================== Subsequent Rounds: Optimize Based on Previous Results ======================
        else:                            
            # Convert images to tensors and extract features
            tensor_fit_images = [preprocess_train(i) for i in fit_images]    
            with torch.no_grad():
                img_embeds = vlmodel.encode_image(torch.stack(tensor_fit_images).to(device))    
            
            # Generate fused images based on image embeddings
            generated_images = fusion_image_to_images(
                Generator, img_embeds, fit_rewards, device, round_save_path, 512
            )
            
            # Create save directory for current round
            fusion_dir = os.path.join(round_save_path, 'fusion')
            os.makedirs(fusion_dir, exist_ok=True)
            
            # Save generated images
            generated_image_paths = []
            for idx, generated_image in enumerate(generated_images):
                image_path = os.path.join(fusion_dir, f'generated_{idx}.jpg')
                generated_image.save(image_path)
                generated_image_paths.append(image_path)
            
            all_viewed_image_paths.extend(generated_image_paths)  # Update viewed image paths
            all_fusion_image_paths.extend(generated_image_paths)
            
            similarities = []
            
            if use_eeg:
                # Send fused images to client and collect ratings and EEG data
                eegs = send_images_and_collect_ratings_and_eeg(generated_image_paths, fusion_dir, len(generated_images))
                
                # Compute similarities and losses for fused images
                for idx, eeg in enumerate(eegs):  
                    if feature_type == 'psd':
                        cs = reward_function(eeg, target_feature, fs, selected_channel_idxes)
                    elif feature_type == 'clip':
                        cs, eeg_feature = reward_function_clip_embed(eeg, eeg_model, target_feature, sub, device)
                    all_viewed_image_rewards.append(cs)
                    all_fusion_image_rewards.append(cs)
                    all_viewed_image_ratings.append(ratings[idx])
                    all_fusion_image_ratings.append(ratings[idx])
                    similarities.append(cs)
                    
                # Update current loop data
                loop_sample_ten.extend(generated_images)
                loop_eeg_ten.extend(eegs)
                loop_reward_ten.extend(similarities)
            else:
                # Send fused images to client and collect ratings
                success = send_images_and_collect_ratings(generated_image_paths, fusion_dir)
                if not success:
                    print("Failed to collect fusion image ratings")
                    return jsonify({"message": "Failed to collect fusion image ratings"}), 500
                    
                # Use user ratings as similarity (reward)
                for rating in ratings:
                    all_viewed_image_rewards.append(rating)
                    all_viewed_image_ratings.append(rating)
                    all_fusion_image_rewards.append(rating)
                    all_fusion_image_ratings.append(rating)
                    similarities.append(rating)
                
                # Clear ratings list
                ratings = []
                
                # Update current loop data
                loop_sample_ten.extend(generated_images)
                loop_reward_ten.extend(similarities)
            
            #====================== Greedy Strategy: Select Images with Similar Features ======================
            greedy_images = []
            sample_image_paths = []
            TOP_K = 10  # Select the K most similar images
            
            # Find the most similar image for each current image embedding
            for img_embed in img_embeds:
                # Find all unprocessed image indices
                available_indices = []
                for i, path in enumerate(test_images_path):
                    if path not in processed_paths:
                        available_indices.append(i)
                        
                if not available_indices:  # Skip if no more unprocessed images
                    continue
                        
                available_features = test_set_img_embeds[available_indices]
                
                # Compute cosine similarity
                cosine_similarities = compute_embed_similarity(img_embed.to(device), available_features.to(device))    
                sorted_available_indices = np.argsort(cosine_similarities.cpu())
                
                # Get top K indices (K highest similarities)
                top_k_count = min(TOP_K, len(sorted_available_indices))
                if top_k_count <= 0:
                    continue
                    
                top_indices = sorted_available_indices[-top_k_count:]
                
                # Randomly select one from top K
                selected_idx = np.random.choice(top_indices)
                actual_idx = available_indices[selected_idx]  # Convert back to original index
                greedy_path = test_images_path[actual_idx]
                greedy_image = Image.open(greedy_path).convert("RGB")
                greedy_images.append(greedy_image)
                sample_image_paths.append(greedy_path)
                    
                # Update processed paths
                processed_paths.add(greedy_path)
            
            # Create save directory for greedy strategy
            greedy_dir = os.path.join(round_save_path, 'greedy')
            os.makedirs(greedy_dir, exist_ok=True)
            
            # Save greedy-selected images
            greedy_image_paths = []
            for idx, greedy_image in enumerate(greedy_images):
                image_path = os.path.join(greedy_dir, f'greedy_{idx}.jpg')
                greedy_image.save(image_path)
                greedy_image_paths.append(image_path)
                
            all_viewed_image_paths.extend(greedy_image_paths)  # Update viewed image paths
            all_greedy_image_paths.extend(greedy_image_paths)
            
            similarities = []
            
            if use_eeg and greedy_images:
                # Send greedy-selected images to client and collect ratings and EEG data
                greedy_eegs = send_images_and_collect_ratings_and_eeg(greedy_image_paths, greedy_dir, len(greedy_images))
                
                # Compute similarities and losses for greedy images
                for idx, eeg in enumerate(greedy_eegs):  
                    if feature_type == 'psd':
                        cs = reward_function(eeg, target_feature, fs, selected_channel_idxes)
                    elif feature_type == 'clip':
                        cs, eeg_feature = reward_function_clip_embed(eeg, eeg_model, target_feature, sub, device)
                    all_viewed_image_rewards.append(cs)
                    all_greedy_image_rewards.append(cs)
                    all_viewed_image_ratings.append(ratings[idx])
                    all_greedy_image_ratings.append(ratings[idx])
                    similarities.append(cs)
                
                # Update current loop data
                loop_sample_ten.extend(greedy_images)
                loop_eeg_ten.extend(greedy_eegs)
                loop_reward_ten.extend(similarities)
            elif greedy_images:  # Only execute when there are greedy images and EEG is not used
                # Send greedy-selected images to client and collect ratings
                success = send_images_and_collect_ratings(greedy_image_paths, greedy_dir)
                if not success:
                    print("Failed to collect greedy image ratings")
                    return jsonify({"message": "Failed to collect greedy image ratings"}), 500
                    
                # Use user ratings as similarity (reward)
                for rating in ratings:
                    all_viewed_image_rewards.append(rating)
                    all_viewed_image_ratings.append(rating)
                    all_greedy_image_rewards.append(rating)
                    all_greedy_image_ratings.append(rating)
                    similarities.append(rating)
                
                # Clear ratings list
                ratings = []
                
                # Update current loop data
                loop_sample_ten.extend(greedy_images)
                loop_reward_ten.extend(similarities)
            
            # Compute selection probabilities based on rewards
            loop_probabilities = softmax(loop_reward_ten)
            
            if use_eeg:
                # Select the best samples from current loop
                chosen_rewards, chosen_images, chosen_eegs = select_from_images(
                    loop_probabilities, loop_reward_ten, loop_sample_ten, loop_eeg_ten, size=4
                )
                
                # Sort the four lists by chosen_rewards in descending order
                combined = list(zip(chosen_rewards, chosen_images, chosen_eegs))
                combined.sort(reverse=True, key=lambda x: x[0])  # Sort by rewards descending

                # Unzip sorted data
                chosen_rewards, chosen_images, chosen_eegs = zip(*combined)

                # Convert results back to lists
                chosen_rewards = list(chosen_rewards)
                chosen_images = list(chosen_images)
                chosen_eegs = list(chosen_eegs)
            else:
                # Select the best samples from current loop (without EEG)
                chosen_rewards, chosen_images = select_from_images_without_eeg(
                    loop_probabilities, loop_reward_ten, loop_sample_ten, size=4
                )
                
                # Sort
                combined = list(zip(chosen_rewards, chosen_images))
                combined.sort(reverse=True, key=lambda x: x[0])
                
                # Unzip sorted data
                chosen_rewards, chosen_images = zip(*combined)
                
                # Convert results back to lists
                chosen_rewards = list(chosen_rewards)
                chosen_images = list(chosen_images)

                    # Update pseudo target model
                        # Add data to pseudo target model
            tensor_loop_sample_ten = [preprocess_train(i) for i in loop_sample_ten]    
            with torch.no_grad():
                tensor_loop_sample_ten_embeds = vlmodel.encode_image(torch.stack(tensor_loop_sample_ten).to(device))        
            

            Generator.pseudo_target_model.add_model_data(
                torch.tensor(tensor_loop_sample_ten_embeds).to(device), 
                (-torch.tensor(loop_reward_ten) * Generator.reward_scaling_factor).to(device)
            )
        # Update fitting data
        fit_images = chosen_images
        fit_rewards = chosen_rewards
        if use_eeg:
            fit_eegs = chosen_eegs
        
        # Update global data records
        all_chosen_rewards.extend(chosen_rewards)
        all_chosen_images.extend(chosen_images)
        if use_eeg:
            all_chosen_eegs.extend(chosen_eegs)
        

        
        # Visualize the highest-rated images in current loop
        visualize_top_images(loop_sample_ten, loop_reward_ten, output_save_path, t)

        # Record and update historical best similarity
        max_similarity = max(loop_reward_ten)
        max_index = loop_reward_ten.index(max_similarity)
        
        if len(history_cs) == 0:
            history_cs.append(max_similarity)
        else:
            max_history = max(history_cs)
            if max_similarity > max_history:
                history_cs.append(max_similarity)
            else:
                history_cs.append(max_history)

        # Convergence check: terminate early if similarity change between consecutive rounds is very small
        if len(history_cs) >= 2:
            if history_cs[-1] != history_cs[-2]:
                diff = abs(history_cs[-1] - history_cs[-2])
                print(history_cs[-1], history_cs[-2], diff)
                if diff <= 1e-4:
                    print("The difference is within 10e-4, stopping.")
                    break
    
    viewed_paths_array = np.array(all_viewed_image_paths, dtype=object)
    save_viewed_paths = os.path.join(output_save_path, 'viewed_image_paths.npy')
    np.save(save_viewed_paths, viewed_paths_array)
    print(f"All viewed image paths saved to: {save_viewed_paths}")
    print(f"Subject viewed a total of {len(all_viewed_image_paths)} images")
    
    # Save all rewards
    all_viewed_image_rewards_array = np.array(all_viewed_image_rewards, dtype=object)
    save_rewards_path = os.path.join(output_save_path, 'all_viewed_image_rewards.npy')
    np.save(save_rewards_path, all_viewed_image_rewards_array)
    print(f"All viewed image reward values saved to: {save_rewards_path}")
    print(f"rewards length: {len(all_viewed_image_rewards)}")
    
    # Save all ratings
    all_viewed_image_ratings_array = np.array(all_viewed_image_ratings, dtype=object)
    save_ratings_path = os.path.join(output_save_path, 'all_viewed_image_ratings.npy')
    np.save(save_ratings_path, all_viewed_image_ratings_array)
    print(f"All viewed image ratings saved to: {save_ratings_path}")
    print(f"ratings length: {len(all_viewed_image_ratings)}")
    
    # Save greedy-selected image paths
    greedy_paths_array = np.array(all_greedy_image_paths, dtype=object)
    save_greedy_paths = os.path.join(output_save_path, 'greedy_image_paths.npy')
    np.save(save_greedy_paths, greedy_paths_array)
    print(f"All greedy-selected image paths saved to: {save_greedy_paths}")
    print(f"Greedy image count: {len(all_greedy_image_paths)}")
    
    # Save fusion-generated image paths
    fusion_paths_array = np.array(all_fusion_image_paths, dtype=object)
    save_fusion_paths = os.path.join(output_save_path, 'fusion_image_paths.npy')
    np.save(save_fusion_paths, fusion_paths_array)
    print(f"All fusion-generated image paths saved to: {save_fusion_paths}")
    print(f"Fusion image count: {len(all_fusion_image_paths)}")
    
    # Save greedy-selected image reward values
    greedy_rewards_array = np.array(all_greedy_image_rewards, dtype=object)
    save_greedy_rewards = os.path.join(output_save_path, 'greedy_image_rewards.npy')
    np.save(save_greedy_rewards, greedy_rewards_array)
    print(f"All greedy-selected image reward values saved to: {save_greedy_rewards}")
    print(f"Greedy reward count: {len(all_greedy_image_rewards)}")
    
    # Save fusion-generated image reward values
    fusion_rewards_array = np.array(all_fusion_image_rewards, dtype=object)
    save_fusion_rewards = os.path.join(output_save_path, 'fusion_image_rewards.npy')
    np.save(save_fusion_rewards, fusion_rewards_array)
    print(f"All fusion-generated image reward values saved to: {save_fusion_rewards}")
    print(f"Fusion reward count: {len(all_fusion_image_rewards)}")
    
    # Save greedy-selected image ratings
    greedy_ratings_array = np.array(all_greedy_image_ratings, dtype=object)
    save_greedy_ratings = os.path.join(output_save_path, 'greedy_image_ratings.npy')
    np.save(save_greedy_ratings, greedy_ratings_array)
    print(f"All greedy-selected image ratings saved to: {save_greedy_ratings}")
    print(f"Greedy rating count: {len(all_greedy_image_ratings)}")
    
    # Save fusion-generated image ratings
    fusion_ratings_array = np.array(all_fusion_image_ratings, dtype=object)
    save_fusion_ratings = os.path.join(output_save_path, 'fusion_image_ratings.npy')
    np.save(save_fusion_ratings, fusion_ratings_array)
    print(f"All fusion-generated image ratings saved to: {save_fusion_ratings}")
    print(f"Fusion rating count: {len(all_fusion_image_ratings)}")
    
    # Output experiment result statistics
    print(f"chosen_rewards {len(chosen_rewards)}")
    print(f"all_chosen_images {len(all_chosen_images)}")
    if use_eeg:
        print(f"all_chosen_eegs {len(all_chosen_eegs)}")

    # Plot similarity history chart
    plt.figure(figsize=(10, 5))
    plt.plot(history_cs, marker='o', markersize=3, label='Similarity')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend() 
    fig_path = os.path.join(output_save_path, 'similarities.jpg')
    plt.savefig(fig_path)
    
    # Notify client that experiment is complete
    socketio.emit('experiment_finished', {
        "message": "Experiment completed"
    })
    
    # Return experiment results
    return jsonify({
        "message": "Experiment completed successfully",
    }), 200
    
@app.route('/experiment_2_post', methods=['POST'])
def experiment_2_post():
    """Present returned offline stimuli to the subject"""
    global selected_channel_idxes
    global target_eeg_path 
    global target_image_path
    global target_feature
    global ratings
    
    print("\n" + "#" * 50)
    print("Collecting Offline Stimuli")
    print("#" * 50 + "\n")
    
    # Search for images
    offline_images = [f for f in os.listdir(offline_images_path) if f.endswith('.jpg') or f.endswith('.png')]
    print(f"Total {len(offline_images)} images")
    offline_images_path = [os.path.join(offline_images_path, offline_image) for offline_image in offline_images]
    
    # Build image path list
    all_viewed_image_paths = offline_images_path.copy()  # Track all viewed image paths
    all_viewed_image_rewards = []
    all_viewed_image_ratings = []
    
    # Create save directory
    offline_save_path = os.path.join(output_save_path, f'offline')
    shutil.rmtree(offline_save_path, ignore_errors=True)  # Clear previous output
    os.makedirs(offline_save_path, exist_ok=True)
    
    eegs = send_images_and_collect_ratings_and_eeg(offline_images_path, offline_save_path, len(offline_images_path))
    
    for idx, eeg in enumerate(eegs):
        if feature_type == 'psd':
            cs = reward_function(eeg, target_feature, fs, selected_channel_idxes)
        elif feature_type == 'clip':
            cs, eeg_feature = reward_function_clip_embed(eeg, eeg_model, target_feature, sub, device)
            
        print(f"reward: {cs}")
        print(f"rating: {ratings[idx]}")
        all_viewed_image_rewards.append(cs)
        all_viewed_image_ratings.append(ratings[idx])
        
    viewed_paths_array = np.array(all_viewed_image_paths, dtype=object)
    save_viewed_paths = os.path.join(output_save_path, 'viewed_image_paths.npy')
    np.save(save_viewed_paths, viewed_paths_array)
    print(f"All viewed image paths saved to: {save_viewed_paths}")
    print(f"Subject viewed a total of {len(all_viewed_image_paths)} images")
    
    # Save all rewards
    all_viewed_image_rewards_array = np.array(all_viewed_image_rewards, dtype=object)
    save_rewards_path = os.path.join(output_save_path, 'all_viewed_image_rewards.npy')
    np.save(save_rewards_path, all_viewed_image_rewards_array)
    print(f"All viewed image reward values saved to: {save_rewards_path}")
    print(f"rewards length: {len(all_viewed_image_rewards)}")
    
    # Save all ratings
    all_viewed_image_ratings_array = np.array(all_viewed_image_ratings, dtype=object)
    save_ratings_path = os.path.join(output_save_path, 'all_viewed_image_ratings.npy')
    np.save(save_ratings_path, all_viewed_image_ratings_array)
    print(f"All viewed image ratings saved to: {save_ratings_path}")
    print(f"ratings length: {len(all_viewed_image_ratings)}")
        
    
    socketio.emit('experiment_finished', {
        "message": "Experiment completed"
    })

@app.route('/eeg_upload', methods=['POST'])
def receive_eeg():
    """Handle EEG data uploaded by the client"""
    try:
        global eeg
        global eeg_received_event
        
        print("Receiving EEG data...")
        
        # Validate request contains files
        if 'files' not in request.files:
            print("Error: No 'files' field in request")
            return jsonify({"message": "No file uploaded, please ensure the request contains a 'file' field"}), 400
        
        file = request.files['files']
        if file.filename == '':
            print("Error: Filename is empty")
            return jsonify({"message": "Filename is empty"}), 400
        
        file_content = file.read()
        file_like_object = io.BytesIO(file_content)
        eeg = np.load(file_like_object)
        print(f"/eeg_upload: EEG data shape: {eeg.shape}")

        # Set event to notify waiting functions to continue
        eeg_received_event.set()
        
        return jsonify({"message": "EEG data received successfully"}), 200
    
    except Exception as e:
        print(f"Error processing EEG upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"message": f"Error processing EEG upload: {str(e)}"}), 500

@app.route('/rating_upload', methods=['POST'])
def receive_ratings():
    """Handle rating data uploaded by the client"""
    global ratings
    global rating_received_event
    
    # Get rating data from request
    data = request.get_json()
    ratings = data.get('ratings', [])
    
    # Ensure cache directory exists
    os.makedirs(cache_path, exist_ok=True)
    
    # Save ratings to cache directory
    save_path = os.path.join(cache_path, 'ratings.json')
    with open(save_path, 'w') as f:
        json.dump(ratings, f, indent=4)
    
    # Set event to notify waiting functions to continue
    rating_received_event.set()
    
    return jsonify({"message": "Ratings received successfully"}), 200

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection event"""
    print('Client disconnected')

def send_images_and_collect_ratings(image_paths, save_path):
    """
    Send images to client and collect ratings.
    
    Args:
        image_paths: List of image file paths
        save_path: Directory path to save ratings
        
    Returns:
        bool: Whether the operation was successful
    """
    global rating_received_event
    global ratings
    
    # Reset event state
    rating_received_event.clear()    
    
    # Encode images and send to client
    print("Sending images to client")
    images = []
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            images.append(encoded_string)
    socketio.emit('image_for_rating', {'images': images})
    
    # Wait for rating received event
    print("Waiting for client ratings...")
    rating_received_event.wait(timeout=300)  # Set timeout (seconds) to avoid infinite wait
    
    # Check if ratings were received
    if not rating_received_event.is_set():
        print("Warning: Rating collection timed out")
        return False
    
    print("Ratings received, continuing execution")
    
    # Print ratings
    print(f"Received ratings: {ratings}")
    
    # Save ratings to specified path
    ratings_file = os.path.join(save_path, 'ratings.json')
    with open(ratings_file, 'w') as f:
        json.dump(ratings, f, indent=4) 
    
    return True    

def send_images_and_collect_ratings_and_eeg(image_paths, save_dir, num_of_events):
    """
    Send images to client and simultaneously collect ratings and EEG data,
    then segment the EEG into the specified number of events.
    
    Args:
        image_paths: List of image file paths
        save_dir: Directory path to save data
        label: Data label used for generating filenames
        
    Returns:
        bool: Whether the operation was successful
    """
    global ratings 
    global eeg
    global eeg_received_event
    global rating_received_event
    
    # Reset event states
    eeg_received_event.clear()
    rating_received_event.clear()
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Encode images and send to client
    print("Sending images to client")
    images = []
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            images.append(encoded_string)
    socketio.emit('image_for_rating_and_eeg', {'images': images})
    
    # Wait for ratings and EEG data
    print("Waiting for client ratings and EEG data...")
    timeout = 300  # Set timeout (seconds)
    
    # Wait for both events
    start_time = time.time()
    while time.time() - start_time < timeout:
        if eeg_received_event.is_set() and rating_received_event.is_set():
            print("Ratings and EEG data received")
            break
        time.sleep(0.5)  # Avoid excessive CPU usage
    
    # Check if all data was received
    if not (eeg_received_event.is_set() and rating_received_event.is_set()):
        print("Warning: Timed out waiting for ratings or EEG data")
        return False
    
    # Save ratings
    ratings_file = os.path.join(save_dir, f'ratings.json')
    with open(ratings_file, 'w') as f:
        json.dump(ratings, f, indent=4)
    
    # Process EEG data: segment, filter, and resample
    event_data_list = create_n_event_npy(eeg, num_of_events)
    filters = prepare_filters(fs, new_fs=250)
    processed_event_data_list = []
    for idx, event_data in enumerate(event_data_list):
        data = real_time_process(event_data, filters)
        if (feature_type == 'clip'):
            data = convert_eeg(data)
        processed_event_data_list.append(data)
        eeg_file = os.path.join(save_dir, f'eeg_{idx}.npy')
        np.save(eeg_file, data)
        
        
    print(f"Data saved to {save_dir}")
    return processed_event_data_list

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=45525)