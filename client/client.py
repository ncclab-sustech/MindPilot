import base64
from io import BytesIO
import requests
import os
import numpy as np
import json
from PIL import Image
import time
import pygame as pg
from pygame_utils import EEGModel, BaseModel, View, EEGController, BaseController
import socketio
import shutil

pre_eeg_path = f'client\pre_eeg'
instant_eeg_path = f'client\instant_eeg'
instant_image_path = f'client\data\instant_image'
image_set_path = f'stimuli_SX' 
    

selected_channels = []
use_eeg= True

url = 'http://10.20.97.33:45525'

sio = socketio.Client() 

# 建立连接
# @sio.event
# def connect():
#     print('Connected to server')
#     time.sleep(1)

@sio.event
def experiment_2_ready():
    view.display_text('Experiment is ready, please wait')
    time.sleep(4)
    # # 向服务器发送开始实验的信号
    send_url = f'{url}/experiment_2'
    requests.post(send_url)

@sio.event
def experiment_1_ready():
    view.display_text('Experiment is ready, please wait')
    time.sleep(4)
    # # 向服务器发送开始实验的信号
    send_url = f'{url}/experiment_1'
    requests.post(send_url)

def send_files_to_server(pre_eeg_path, url):
    files = []
    file_objects = []

    # 遍历 pre_eeg_path 中的所有 .npy 文件，并发送
    try:
        for filename in os.listdir(pre_eeg_path):
            if filename.endswith('.npy'):
                file_path = os.path.join(pre_eeg_path, filename)
                f = open(file_path, 'rb')
                file_objects.append(f)
                files.append(('files', (filename, f, 'application/octet-stream')))

        response = requests.post(url, files=files)
        print("Files sent successfully")
    finally:
        # 确保所有文件在请求完成后被关闭
        for f in file_objects:
            f.close()
            
@sio.event
def image_for_rating(data):
    os.makedirs(instant_image_path, exist_ok=True)
    shutil.rmtree(instant_image_path)
    print('Images received')
    images = data['images']
    for idx, encoded_string in enumerate(images):
        image_data = base64.b64decode(encoded_string)
        image = Image.open(BytesIO(image_data))
        # 保存图像到 client/data/instant_image 目录下
        image_save_path = os.path.join(instant_image_path, f'image_{idx}.png')
        os.makedirs(instant_image_path, exist_ok=True)
        image.save(image_save_path)
        print(f'Image saved to {image_save_path}')
        
    print('All images saved')
    
    # 启动实验
    ratings = controller.start_rating(instant_image_path)
    
    # 发送 ratings 到服务器
    send_url = f'{url}/rating_upload'
    headers = {'Content-Type': 'application/json'}
    # 确保 ratings 是一个 JSON 数组
    data = {
        'ratings': list(map(float, ratings))  # 确保 ratings 是浮点数列表
    }
    response = requests.post(send_url, headers=headers, json=data)  # 使用 json 参数直接传递数据
    print('Ratings sent to server:', response.status_code, response.text)

    # 删除 instant_image_path 中的所有文件
    shutil.rmtree(instant_image_path)
    
@sio.event
def image_for_rating_and_eeg(data):
    os.makedirs(instant_image_path, exist_ok=True)
    os.makedirs(instant_eeg_path, exist_ok=True)
    os.makedirs(instant_image_path, exist_ok=True)
    os.makedirs(instant_eeg_path, exist_ok=True)
    # 删除 instant_image_path 和 instant_eeg_path 中的所有文件
    shutil.rmtree(instant_image_path)
    shutil.rmtree(instant_eeg_path)    
    print('Images received')
    images = data['images']
    for idx, encoded_string in enumerate(images):
        image_data = base64.b64decode(encoded_string)
        image = Image.open(BytesIO(image_data))
        # 保存图像到 client/data/instant_image 目录下
        image_save_path = os.path.join(instant_image_path, f'image_{idx}.png')
        os.makedirs(instant_image_path, exist_ok=True)
        image.save(image_save_path)
        print(f'Image saved to {image_save_path}')
    
    print('All images saved')

    # 启动实验
    ratings = controller.start_collect_and_rating(instant_image_path, instant_eeg_path)
    
    # 发送 ratings 到服务器
    send_url = f'{url}/rating_upload'
    headers = {'Content-Type': 'application/json'}
    # 确保 ratings 是一个 JSON 数组
    data = {
        'ratings': list(map(float, ratings))  # 确保 ratings 是浮点数列表
    }
    response = requests.post(send_url, headers=headers, json=data)  # 使用 json 参数直接传递数据
    print('Ratings sent to server:', response.status_code, response.text)
    
    # 发送 instant_eeg_path 中的所有 npy 文件到服务器
    send_url = f'{url}/eeg_upload'
    send_files_to_server(instant_eeg_path, send_url)
    
    # 删除 instant_image_path 中的所有文件
    shutil.rmtree(instant_image_path)
    
    
@sio.event
def image_for_collection(data):
    os.makedirs(instant_image_path, exist_ok=True)
    os.makedirs(instant_eeg_path, exist_ok=True)
    # 删除 instant_image_path 和 instant_eeg_path 中的所有文件
    shutil.rmtree(instant_image_path)
    shutil.rmtree(instant_eeg_path)    
    print('Images received')
    images = data['images']
    for idx, encoded_string in enumerate(images):
        image_data = base64.b64decode(encoded_string)
        image = Image.open(BytesIO(image_data))
        # 保存图像到 client/data/instant_image 目录下
        image_save_path = os.path.join(instant_image_path, f'image_{idx}.png')
        os.makedirs(instant_image_path, exist_ok=True)
        image.save(image_save_path)
        print(f'Image saved to {image_save_path}')
    
    print('All images saved')

    # 启动实验
    controller.start_collection(instant_image_path, instant_eeg_path)
    
    # 发送 instant_eeg_path 中的所有 npy 文件到服务器
    send_url = f'{url}/instant_eeg_upload'
    send_files_to_server(instant_eeg_path, send_url)


@sio.event
def experiment_finished(data):
    print(data['message'])
    if use_eeg:
        controller.stop_collection()
    # 断开连接
    sio.disconnect()
    quit


if __name__ == '__main__':
    global controller
    
    if (use_eeg):
        model = EEGModel()
        view = View()
        controller = EEGController(model, view)
    else:
        model = BaseModel()
        view = View()
        controller = BaseController(model, view)


    sio.connect(url) 

    controller.run()

    # 等待以保持连接
    try:
        sio.wait()
    except KeyboardInterrupt:
        sio.disconnect()