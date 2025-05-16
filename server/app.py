import base64
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import os
import random
import shutil
import time

from modulation_utils import *
from modulation import fusion_image_to_images

app = Flask(__name__)
socketio = SocketIO(app)

# 路径参数
image_set_path = 'stimuli_SX'
pre_eeg_path = 'server/pre_eeg' # TODO:修改！
instant_eeg_path = 'server/instant_eeg'

# 实验参数
num_loop_random = 1
subject_id = 1 
num_loops = 10
sub = 'sub-' + (str(subject_id) if subject_id >= 10 else format(subject_id, '02')) # 如果 subject_id 大于或等于 10，直接使用其值；如果小于 10，则将其格式化为两位数字（如 01, 02）。
fs = 250

# 全局变量
selected_channel_idxes = None
target_image_path = None
target_eeg_path = None

@socketio.on('connect')
def handle_connect(auth):
    print('Client connected')
    print('Send: pre_experiment_ready')
    socketio.emit('pre_experiment_ready')

@app.route('/pre_experiment_eeg_upload', methods=['POST'])
def pre_experiment():
    global selected_channel_idxes
    global target_image_path
    global target_eeg_path
    if 'files' not in request.files:
        return jsonify({"message": "No file part"}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({"message": "No selected files"}), 400

    for file in files:
        if file.filename == '':
            return jsonify({"message": "No selected file"}), 400
        if file:
            filename = file.filename
            save_path = os.path.join(pre_eeg_path, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            file.save(save_path)

    # 载入保存的文件
    file_list = [os.path.join(pre_eeg_path, f) for f in sorted(os.listdir(pre_eeg_path))]
    eeg_data = np.array([np.load(file) for file in file_list])  # (n_samples, n_channels, n_timepoints)
    
    # 运行 get_selected_channel_idxes 和 get_target_image 函数
    selected_channel_idxes = get_selected_channel_idxes(eeg_data, fs)
    target_image_index = get_target_image_index(eeg_data, selected_channel_idxes, fs)
    print("Selected channels:", selected_channel_idxes)
    print("Target image index:", target_image_index)

    # 找到第 target_image_index 个文件夹里的图片
    folder_list = sorted(os.listdir(image_set_path))
    folder_path = os.path.join(image_set_path, folder_list[target_image_index])
    target_image_path = os.path.join(folder_path, os.listdir(folder_path)[0])
    target_eeg_path = f'{pre_eeg_path}/{target_image_index}.npy'
    print("Target image path:", target_image_path)
    print("Target eeg path:", target_eeg_path)

    # 向客户端发送 experiment_ready 信号
    print('Send: experiment_ready')
    socketio.emit('experiment_ready') 

    return jsonify({
        "message": f"Files uploaded successfully"
    }), 200
        

@app.route('/experiment', methods=['POST'])
def experiment():
    global selected_channel_idxes
    global target_image_path
    global target_eeg_path

    print("\n" + "#" * 50)
    print("🚀 Starting Experiment 🚀")
    print(f"🎯 Target Image Path: {target_image_path}")
    print(f"📊 Target EEG Path: {target_eeg_path}")
    print("#" * 50 + "\n")

    time.sleep(1)

    base_seed = 100000 * subject_id
    for i in range(1, num_loop_random + 1):
        base_save_path = f'/mnt/dataset0/xkp/closed-loop/exp_sub{subject_id}/loop_random_{i}'
        os.makedirs(base_save_path, exist_ok=True)
        seed = base_seed + i 
        np.random.seed(seed)
        random.seed(seed)
        print(f'Subject {subject_id}- Run {i}/{num_loop_random} - Seed: {seed}')
        target_psd = load_target_psd(target_eeg_path, fs, selected_channel_idxes)
        test_images_path, _ = get_image_pool(image_set_path)
        test_images_path.remove(target_image_path)

        processed_paths = set()
        
        all_chosen_similarities = []
        all_chosen_losses = []
        all_chosen_image_paths = []
        all_chosen_eeg_paths = []
        history_cs = []
        history_loss = []

        for i in range(num_loops):
            print(f"Loop {i + 1}/{num_loops}")
            round_save_path = os.path.join(base_save_path, f'loop{i + 1}')
            loop_sample_ten = []
            loop_cs_ten = []
            loop_eeg_ten = []
            loop_loss_ten = []
            os.makedirs(base_save_path, exist_ok=True)

            if i == 0:
                first_ten = os.path.join(round_save_path, 'first_ten')
                os.makedirs(first_ten, exist_ok=True)
                available_paths = [path for path in test_images_path if path not in processed_paths]
                sample_image_paths = sorted(random.sample(available_paths, 10))
                processed_paths.update(sample_image_paths)
                sample_image_name = []
                for sample_image_path in sample_image_paths:
                    filename = os.path.basename(sample_image_path).split('.')[0]
                    sample_image_name.append(filename)
                collect_and_save_eeg_for_all_images(sample_image_paths, first_ten, sample_image_name)

                similarities = []
                sample_eeg_paths = []
                losses = []
                for eeg in sorted(os.listdir(first_ten)):
                    filename = eeg.split('.')[0]
                    eeg_path = os.path.join(first_ten, eeg)
                    sample_eeg_paths.append(eeg_path)
                    cs = calculate_similarity(eeg_path, target_psd, fs, selected_channel_idxes)
                    similarities.append(cs)
                    loss = calculate_loss(eeg_path, target_psd, fs, selected_channel_idxes)
                    losses.append(loss)
                probabilities = softmax(similarities)
                chosen_similarities, chosen_losses, chosen_image_paths, chosen_eeg_paths = select(probabilities, similarities, losses, sample_image_paths, sample_eeg_paths)                
            else:
                chosen_similarities = all_chosen_similarities[-2:]
                chosen_image_paths = all_chosen_image_paths[-2: ]
            loop_sample_ten = [chosen_image_path for chosen_image_path in chosen_image_paths]
            loop_cs_ten = [chosen_similarity for chosen_similarity in chosen_similarities]
            loop_eeg_ten = [chosen_eeg_path for chosen_eeg_path in chosen_eeg_paths]
            loop_loss_ten = [chosen_loss for chosen_loss in chosen_losses]
            # print(chosen_similarities, chosen_losses, chosen_image_paths, chosen_eeg_paths)
            fusion_image_to_images(chosen_image_paths, 6, round_save_path, 256)
            image_path_list = []
            label_list = []
            for image in sorted(os.listdir(round_save_path)):
                if image.endswith('jpg'):
                    image_path = os.path.join(round_save_path, image)
                    loop_sample_ten.append(image_path)
                    image_path_list.append(image_path)
                    file_name = os.path.splitext(image)[0]
                    label_list.append(file_name)
            collect_and_save_eeg_for_all_images(image_path_list, round_save_path, label_list)
            for eeg in sorted(os.listdir(round_save_path)):
                if eeg.endswith('npy'):
                    eeg_path = os.path.join(round_save_path, eeg)
                    loop_eeg_ten.append(eeg_path)
                    cs = calculate_similarity(eeg_path, target_psd, fs, selected_channel_idxes)
                    loss = calculate_loss(eeg_path, target_psd, fs, selected_channel_idxes)
                    loop_cs_ten.append(cs)
                    loop_loss_ten.append(loss)
            available_paths = [path for path in test_images_path if path not in processed_paths]
            # print(len(available_paths))
            sample_image_paths = sorted(random.sample(available_paths, min(2, len(available_paths))))
            new_sample_list = []
            new_sample_label = []
            for sample_image_path in sample_image_paths:
                filename = os.path.basename(sample_image_path).split('.')[0]
                new_sample_label.append(filename)
                loop_sample_ten.append(sample_image_path)
                new_sample_list.append(sample_image_path)
            # print(loop_sample_ten)
            processed_paths.update(sample_image_paths)
            # print(len(processed_paths))
            new_save_path = os.path.join(round_save_path, 'new_sample_signal')
            os.makedirs(new_save_path, exist_ok=True)
            collect_and_save_eeg_for_all_images(new_sample_list, new_save_path, new_sample_label)
            for new_sample_eeg in sorted(os.listdir(new_save_path)):
                new_sample_eeg_path = os.path.join(new_save_path, new_sample_eeg)
                loop_eeg_ten.append(new_sample_eeg_path)
                cs = calculate_similarity(new_sample_eeg_path, target_psd, fs, selected_channel_idxes)
                loss = calculate_loss(new_sample_eeg_path, target_psd, fs, selected_channel_idxes)
                loop_cs_ten.append(cs)
                loop_loss_ten.append(loss)
            # print(loop_cs_ten)
            # print(loop_loss_ten)
            probabilities = softmax(loop_cs_ten)
            # print(probabilities)

            chosen_similarities, chosen_losses, chosen_image_paths, chosen_eeg_paths = select(probabilities, loop_cs_ten, loop_loss_ten, loop_sample_ten,loop_eeg_ten)
            
            # print("Chosen similarities:", chosen_similarities)
            # print("Chosen losses:", chosen_losses)
            # print("Chosen image paths:", chosen_image_paths)
            # print("Chosen eeg paths:", chosen_eeg_paths)

            for chosen_similarity in chosen_similarities:
                all_chosen_similarities.append(chosen_similarity)
            for chosen_loss in chosen_losses:
                all_chosen_losses.append(chosen_loss)
            for chosen_image_path in chosen_image_paths:
                all_chosen_image_paths.append(chosen_image_path)
            for chosen_eeg_path in chosen_eeg_paths:
                all_chosen_eeg_paths.append(chosen_eeg_path)      

            max_similarity = max(chosen_similarities)
            # print('max_similarity:', max_similarity)
            max_index = chosen_similarities.index(max_similarity)
            corresponding_loss = chosen_losses[max_index]
            # print("Corresponding loss:", corresponding_loss)

            if len(history_cs) == 0:
                history_cs.append(max_similarity)
                history_loss.append(corresponding_loss) 
            else:
                max_history = max(history_cs)
                if max_similarity > max_history:
                    history_cs.append(max_similarity)
                    history_loss.append(corresponding_loss)
                else:
                    history_cs.append(max_history)
                    history_loss.append(history_loss[-1])

            print(history_cs)
            print(history_loss)

            if len(history_cs) >= 2:
                if history_cs[-1] != history_cs[-2]:
                    diff = abs(history_cs[-1] - history_cs[-2])
                    print(history_cs[-1], history_cs[-2], diff)
                    if diff <= 1e-4:
                        print("The difference is within 10e-4, stopping.")
                        break

        print(all_chosen_similarities)
        print(all_chosen_losses)
        print(all_chosen_image_paths)
        print(all_chosen_eeg_paths)

    print("Sending end signal to client")
    socketio.emit('experiment_finished', {'message': 'Experiment finished'})


@app.route('/instant_eeg_upload', methods=['POST'])
def process_instant_eeg():
    if 'files' not in request.files:
        return jsonify({"message": "No file part"}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({"message": "No selected files"}), 400

    for file in files:
        if file.filename == '':
            return jsonify({"message": "No selected file"}), 400
        if file:
            filename = file.filename
            save_path = os.path.join(instant_eeg_path, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            file.save(save_path)

    return jsonify({
        "message": f"Files uploaded successfully"
    }), 200


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


def collect_and_save_eeg_for_all_images(image_paths, save_path, category_list):
    print("Sending images to client")
    images = []
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            images.append(encoded_string)
    socketio.emit('images_received', {'images': images})

    os.makedirs(instant_eeg_path, exist_ok=True)

    while True:
        files = [f for f in os.listdir(instant_eeg_path) if f.endswith('.npy')]
        if files:
            break
        else:
            time.sleep(1)

    time.sleep(10)

    print("Category number:", len(category_list))

    # 遍历 category_list，寻找对应的文件
    for idx, category in enumerate(category_list):
        filename = f"{idx+1}.npy"
        file_path = os.path.join(instant_eeg_path, filename)
        if os.path.exists(file_path):
            new_filename = f"{category}_{filename}"
            dest_path = os.path.join(save_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            shutil.move(file_path, dest_path)
            print(f"Moved and renamed file to {dest_path}")
        else:
            print(f"File {filename} not found in {instant_eeg_path}")

    shutil.rmtree(instant_eeg_path)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=45565)