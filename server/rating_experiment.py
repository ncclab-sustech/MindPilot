import base64
import json
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import joblib
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import random
import shutil
import time
from threading import Event

from modulation_utils import *
from modulation import fusion_image_to_images

from server_utils import *

app = Flask(__name__)
socketio = SocketIO(app)

# 实验参数
subject_id = 1
fs = 250
num_loops = 10
use_eeg = True


# 路径参数
image_set_path = 'stimuli_SX'
pre_eeg_path = f'server/data/sub{subject_id}/pre_eeg' # TODO: 验证
instant_eeg_path = 'server/data/instant_eeg'
cache_path = 'server/data/cache'
target_image_path = 'stimuli_SX/Dis-07.jpg'
target_eeg_path = 'DIRTI Database/1139_body products.jpg'


# 全局变量
selected_channel_idxes = []
target_image_path = None
target_eeg_path = None
clf = None
rating_received_event = Event()
eeg_received_event = Event()
# 临时存储的数据
ratings = []
eeg = None

@socketio.on('connect')
def handle_connect(auth):
    print('Client connected')
    print('Send: experiment_2_ready')
    socketio.emit('experiment_2_ready')

@app.route('/experiment_2', methods=['POST'])
def experiment_2():
    global selected_channel_idxes
    global target_eeg_path
    global target_image_path
    global ratings

    print("\n" + "#" * 50)
    print("图片 rating 迭代实验")
    print("#" * 50 + "\n")
    
    seed = int(10000 * time.time() % 10000)
    print(f"随机种子: {seed}")
    base_save_path = f'server/data/sub{subject_id}/rating_experiment'
    os.makedirs(base_save_path, exist_ok=True)
    
    # 获取所有图片
    test_images = [f for f in os.listdir(image_set_path) if f.endswith('.jpg') or f.endswith('.png')]
    # amu_images = [f for f in all_image_files if f.startswith('Amu-')]
    # dis_images = [f for f in all_image_files if f.startswith('Dis-')]
    # test_images = amu_images + dis_images
    # print(f"找到 {len(amu_images)} 张 Amu 图片")
    # print(f"找到 {len(dis_images)} 张 Dis 图片")
    print(f"一共 {len(test_images)} 张图片")
    
    # 随机所有照片
    random.seed(seed)
    random.shuffle(test_images)
    
    test_images_path = [os.path.join(image_set_path, test_image) for test_image in test_images]
    
    # 如果存在目标图片，则从测试集中移除
    if target_image_path in test_images_path:
        test_images_path.remove(target_image_path)
    
    processed_paths = set()
    
    all_chosen_ratings = []      # 记录所有选中图片的评分
    all_chosen_image_paths = []  # 记录所有选中的图片路径
    history_best_ratings = []    # 记录每一轮的最高评分
    
    # 固定执行指定轮次，不提前终止
    for i in range(num_loops):
        print(f"第 {i+1}/{num_loops} 轮实验")
        round_save_path = os.path.join(base_save_path, f'loop_{i+1}')
        os.makedirs(round_save_path, exist_ok=True)
        
        loop_sample_images = []  # 当前轮次的图片集合
        loop_ratings = []        # 当前轮次的评分集合
        loop_labels = []        # 当前轮次的标签集合
        
        if i == 0:
            # 第一轮随机抽取10张图片
            first_ten_dir = os.path.join(round_save_path, 'first_ten')
            os.makedirs(first_ten_dir, exist_ok=True)
            
            # 从未处理的图片中随机选择10张
            available_paths = [path for path in test_images_path if path not in processed_paths]
            sample_image_paths = sorted(random.sample(available_paths, min(10, len(available_paths))))
            processed_paths.update(sample_image_paths)
            
            # 发送图片并收集评分
            if (use_eeg):
                # 如果使用EEG数据，发送EEG数据和图片
                success = send_images_and_collect_ratings_and_eeg(sample_image_paths, first_ten_dir, 'first_ten')
            else:
                success = send_images_and_collect_ratings(sample_image_paths, first_ten_dir)
            if not success:
                return jsonify({"message": "评分以及采集失败"}), 500
            
            # 使用用户评分作为相似度
            user_ratings = ratings.copy()
            
            # 记录图片和评分
            loop_sample_images = sample_image_paths
            loop_ratings = user_ratings
            
            # 根据评分选择最好的两张图片
            chosen_indices = sorted(range(len(user_ratings)), key=lambda x: user_ratings[x], reverse=True)[:2]
            chosen_ratings = [user_ratings[idx] for idx in chosen_indices]
            chosen_image_paths = [sample_image_paths[idx] for idx in chosen_indices]
            
        else:
            # 非第一轮，使用之前选择的最佳三张图片
            chosen_ratings = all_chosen_ratings[-3:] if len(all_chosen_ratings) >= 3 else all_chosen_ratings
            chosen_image_paths = all_chosen_image_paths[-3:] if len(all_chosen_image_paths) >= 3 else all_chosen_image_paths
            
            # 将已选图片加入当前轮次集合
            loop_sample_images.extend(chosen_image_paths)
            loop_ratings.extend(chosen_ratings)
        
        # 基于当前最佳图片融合生成新图片
        fusion_dir = os.path.join(round_save_path, 'fusion')
        os.makedirs(fusion_dir, exist_ok=True)
        
        try:
            # 使用融合函数生成新图片
            fusion_image_to_images(chosen_image_paths, 4, fusion_dir, 256)
            
            # 获取融合生成的所有图片路径
            fusion_image_paths = []
            for image in sorted(os.listdir(fusion_dir)):
                if image.endswith('.jpg') or image.endswith('.png'):
                    fusion_image_paths.append(os.path.join(fusion_dir, image))
            
            # 发送融合图片并收集评分
            if fusion_image_paths:
                if (use_eeg):
                    # 如果使用EEG数据，发送EEG数据和图片
                    success = send_images_and_collect_ratings_and_eeg(fusion_image_paths, fusion_dir, 'fusion')
                else:
                    # 发送融合图片并收集评分
                    success = send_images_and_collect_ratings(fusion_image_paths, fusion_dir)
                if not success:
                    return jsonify({"message": "融合图片评分收集失败"}), 500
                
                # 获取融合图片的评分
                fusion_ratings = ratings.copy()
                
                # 将融合图片和评分添加到当前轮次集合
                loop_sample_images.extend(fusion_image_paths)
                loop_ratings.extend(fusion_ratings)
        except Exception as e:
            print(f"图片融合失败: {str(e)}")
        
        # 从未处理的图片池中随机选择新图片
        new_samples_dir = os.path.join(round_save_path, 'new_samples')
        os.makedirs(new_samples_dir, exist_ok=True)
        
        available_paths = [path for path in test_images_path if path not in processed_paths]
        if available_paths:
            new_sample_paths = sorted(random.sample(available_paths, min(3, len(available_paths))))
            processed_paths.update(new_sample_paths)
            
            # 发送新图片并收集评分
            if new_sample_paths:
                success = send_images_and_collect_ratings(new_sample_paths, new_samples_dir)
                if not success:
                    return jsonify({"message": "新样本评分收集失败"}), 500
                
                # 获取新图片的评分
                new_ratings = ratings.copy()
                
                # 将新图片和评分添加到当前轮次集合
                loop_sample_images.extend(new_sample_paths)
                loop_ratings.extend(new_ratings)
        
        # 保存当前轮次的所有图片和评分
        all_ratings_file = os.path.join(round_save_path, 'all_ratings.json')
        with open(all_ratings_file, 'w') as f:
            json.dump({
                "image_paths": loop_sample_images,
                "ratings": loop_ratings
            }, f, indent=4)
        
        # 计算选择概率并选择最佳图片
        if loop_ratings:
            chosen_indices = sorted(range(len(loop_ratings)), key=lambda x: loop_ratings[x], reverse=True)[:2]
            chosen_ratings = [loop_ratings[idx] for idx in chosen_indices]
            chosen_image_paths = [loop_sample_images[idx] for idx in chosen_indices]
            
            # 记录选中的图片和评分
            for rating in chosen_ratings:
                all_chosen_ratings.append(rating)
            for image_path in chosen_image_paths:
                all_chosen_image_paths.append(image_path)
            
            # 更新历史最佳评分
            if chosen_ratings:
                max_rating = max(chosen_ratings)
                if not history_best_ratings or max_rating > max(history_best_ratings):
                    history_best_ratings.append(max_rating)
                else:
                    history_best_ratings.append(history_best_ratings[-1])
        
        # 打印当前轮次信息
        print(f"当前轮次选择的图片: {[os.path.basename(p) for p in chosen_image_paths]}")
        print(f"当前轮次选择的评分: {chosen_ratings}")
        print(f"历史最佳评分: {history_best_ratings}")
    
    # 保存实验总结数据
    summary_file = os.path.join(base_save_path, 'experiment_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            "all_chosen_ratings": all_chosen_ratings,
            "all_chosen_image_paths": all_chosen_image_paths,
            "history_best_ratings": history_best_ratings
        }, f, indent=4)
    
    # 生成评分历史图表
    plt.figure(figsize=(10, 6))
    plt.plot(history_best_ratings, marker='o')
    plt.title('评分历史变化')
    plt.xlabel('迭代轮次')
    plt.ylabel('最佳评分')
    plt.grid(True)
    plt.savefig(os.path.join(base_save_path, 'rating_history.png'))
    
    print("实验完成")
    socketio.emit('experiment_finished', {
        "message": "实验完成"
    })
    
    return jsonify({
        "message": "实验成功完成",
        "best_rating": history_best_ratings[-1] if history_best_ratings else 0,
        "best_image": os.path.basename(all_chosen_image_paths[-1]) if all_chosen_image_paths else ""
    }), 200
        
        
@app.route('/eeg_upload', methods=['POST'])
def receive_eeg():
    try:
        global eeg
        global eeg_received_event
        
        print("接收EEG数据...")
        
        if 'files' not in request.files:
            print("错误: 请求中没有'files'字段")
            return jsonify({"message": "没有上传文件，请确保请求包含'file'字段"}), 400
        
        file = request.files['files']
        if file.filename == '':
            print("错误: 文件名为空")
            return jsonify({"message": "文件名为空"}), 400
        
        # 确保cache目录存在
        os.makedirs(cache_path, exist_ok=True)
        
        # 保存文件到cache目录
        file_path = os.path.join(cache_path, 'eeg.npy')
        file.save(file_path)
        print(f"EEG文件已保存到 {file_path}")
        
        try:
            # 读取npy文件到全局变量
            eeg = np.load(file_path)
            print(f"成功加载EEG数据，形状: {eeg.shape}")
        except Exception as e:
            print(f"EEG数据加载失败: {str(e)}")
            return jsonify({"message": f"无法加载EEG数据: {str(e)}"}), 400
        
        # 设置事件，通知等待的函数继续执行
        eeg_received_event.set()
        
        return jsonify({"message": "EEG数据接收成功"}), 200
    
    except Exception as e:
        print(f"处理EEG上传时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"message": f"处理EEG上传时出错: {str(e)}"}), 500
    
    
@app.route('/rating_upload', methods=['POST'])
def receive_ratings():
    global ratings
    global rating_received_event
    
    data = request.get_json()
    ratings = data.get('ratings', [])
    
    # 确保cache目录存在
    os.makedirs(cache_path, exist_ok=True)
    
    # 保存评分到cache目录
    save_path = os.path.join(cache_path, 'ratings.json')
    with open(save_path, 'w') as f:
        json.dump(ratings, f, indent=4)
    
    # 设置事件，通知等待的函数继续执行
    rating_received_event.set()
    
    return jsonify({"message": "评分接收成功"}), 200


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    

def send_images_and_collect_ratings(image_paths, save_path):
    global rating_received_event
    global ratings
    
    # 重置事件状态
    rating_received_event.clear()    
    
    print("发送图片到客户端")
    images = []
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            images.append(encoded_string)
    socketio.emit('image_for_rating', {'images': images})
    
    # 等待评分接收事件
    print("等待客户端评分...")
    rating_received_event.wait(timeout=300)  # 设置超时时间(秒)，避免无限等待
    
    if not rating_received_event.is_set():
        print("警告: 等待评分超时")
        return False
    
    print("已收到评分，继续执行")
    
    # 保存评分到指定路径
    ratings_file = os.path.join(save_path, 'ratings.json')
    with open(ratings_file, 'w') as f:
        json.dump(ratings, f, indent=4) 
        
    
    
    return True    

def send_images_and_collect_ratings_and_eeg(image_paths, save_dir, label):
    global ratings 
    global eeg
    global eeg_received_event
    global rating_received_event
    
    # 重置事件状态
    eeg_received_event.clear()
    rating_received_event.clear()
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    print("发送图片到客户端")
    images = []
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            images.append(encoded_string)
    socketio.emit('image_for_rating_and_eeg', {'images': images, 'label': label})
    
    print("等待客户端评分和EEG数据...")
    timeout = 300  # 设置超时时间(秒)
    
    # 等待两个事件
    start_time = time.time()
    while time.time() - start_time < timeout:
        if eeg_received_event.is_set() and rating_received_event.is_set():
            print("已接收评分和EEG数据")
            break
        time.sleep(0.5)  # 避免过度占用CPU
    
    if not (eeg_received_event.is_set() and rating_received_event.is_set()):
        print("警告: 等待评分或EEG数据超时")
        return False
    
    # 保存评分
    ratings_file = os.path.join(save_dir, f'{label}_ratings.json')
    with open(ratings_file, 'w') as f:
        json.dump(ratings, f, indent=4)
    
    # 处理EEG数据
    event_data_list = create_n_event_npy(eeg, 1)
    filters = prepare_filters(fs, new_fs=250)
    processed_event_data_list = []
    for event_data in event_data_list:
        data = real_time_process(event_data, filters)
        processed_event_data_list.append(data)
        eeg_file = os.path.join(save_dir, f'N.npy')
        np.save(eeg_file, data)
    print(f"数据已保存到 {save_dir}")
    return True
    


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=45565)