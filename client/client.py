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

# Establish connection
# @sio.event
# def connect():
#     print('Connected to server')
#     time.sleep(1)

@sio.event
def experiment_2_ready():
    view.display_text('Experiment is ready, please wait')
    time.sleep(4)
    # # Send start experiment signal to the server
    send_url = f'{url}/experiment_2'
    requests.post(send_url)

@sio.event
def experiment_1_ready():
    view.display_text('Experiment is ready, please wait')
    time.sleep(4)
    # # Send start experiment signal to the server
    send_url = f'{url}/experiment_1'
    requests.post(send_url)

def send_files_to_server(pre_eeg_path, url):
    files = []
    file_objects = []

    # Iterate through all .npy files in pre_eeg_path and send them
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
        # Ensure all files are closed after the request completes
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
        # Save image to client/data/instant_image directory
        image_save_path = os.path.join(instant_image_path, f'image_{idx}.png')
        os.makedirs(instant_image_path, exist_ok=True)
        image.save(image_save_path)
        print(f'Image saved to {image_save_path}')
        
    print('All images saved')
    
    # Start the experiment
    ratings = controller.start_rating(instant_image_path)
    
    # Send ratings to the server
    send_url = f'{url}/rating_upload'
    headers = {'Content-Type': 'application/json'}
    # Ensure ratings is a JSON array
    data = {
        'ratings': list(map(float, ratings))  # Ensure ratings is a list of floats
    }
    response = requests.post(send_url, headers=headers, json=data)  # Pass data directly using the json parameter
    print('Ratings sent to server:', response.status_code, response.text)

    # Delete all files in instant_image_path
    shutil.rmtree(instant_image_path)
    
@sio.event
def image_for_rating_and_eeg(data):
    os.makedirs(instant_image_path, exist_ok=True)
    os.makedirs(instant_eeg_path, exist_ok=True)
    os.makedirs(instant_image_path, exist_ok=True)
    os.makedirs(instant_eeg_path, exist_ok=True)
    # Delete all files in instant_image_path and instant_eeg_path
    shutil.rmtree(instant_image_path)
    shutil.rmtree(instant_eeg_path)    
    print('Images received')
    images = data['images']
    for idx, encoded_string in enumerate(images):
        image_data = base64.b64decode(encoded_string)
        image = Image.open(BytesIO(image_data))
        # Save image to client/data/instant_image directory
        image_save_path = os.path.join(instant_image_path, f'image_{idx}.png')
        os.makedirs(instant_image_path, exist_ok=True)
        image.save(image_save_path)
        print(f'Image saved to {image_save_path}')
    
    print('All images saved')

    # Start the experiment
    ratings = controller.start_collect_and_rating(instant_image_path, instant_eeg_path)
    
    # Send ratings to the server
    send_url = f'{url}/rating_upload'
    headers = {'Content-Type': 'application/json'}
    # Ensure ratings is a JSON array
    data = {
        'ratings': list(map(float, ratings))  # Ensure ratings is a list of floats
    }
    response = requests.post(send_url, headers=headers, json=data)  # Pass data directly using the json parameter
    print('Ratings sent to server:', response.status_code, response.text)
    
    # Send all .npy files in instant_eeg_path to the server
    send_url = f'{url}/eeg_upload'
    send_files_to_server(instant_eeg_path, send_url)
    
    # Delete all files in instant_image_path
    shutil.rmtree(instant_image_path)
    
    
@sio.event
def image_for_collection(data):
    os.makedirs(instant_image_path, exist_ok=True)
    os.makedirs(instant_eeg_path, exist_ok=True)
    # Delete all files in instant_image_path and instant_eeg_path
    shutil.rmtree(instant_image_path)
    shutil.rmtree(instant_eeg_path)    
    print('Images received')
    images = data['images']
    for idx, encoded_string in enumerate(images):
        image_data = base64.b64decode(encoded_string)
        image = Image.open(BytesIO(image_data))
        # Save image to client/data/instant_image directory
        image_save_path = os.path.join(instant_image_path, f'image_{idx}.png')
        os.makedirs(instant_image_path, exist_ok=True)
        image.save(image_save_path)
        print(f'Image saved to {image_save_path}')
    
    print('All images saved')

    # Start the experiment
    controller.start_collection(instant_image_path, instant_eeg_path)
    
    # Send all .npy files in instant_eeg_path to the server
    send_url = f'{url}/instant_eeg_upload'
    send_files_to_server(instant_eeg_path, send_url)


@sio.event
def experiment_finished(data):
    print(data['message'])
    if use_eeg:
        controller.stop_collection()
    # Disconnect
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

    # Wait to keep the connection alive
    try:
        sio.wait()
    except KeyboardInterrupt:
        sio.disconnect()