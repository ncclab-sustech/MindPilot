import random
import os
import shutil
import numpy as np
import pygame as pg
import time
import gc

from neuracle_api import DataServerThread
from triggerBox import TriggerBox, PackageSensorPara
from eeg_process import *

class BaseModel:
    pass
        

class EEGModel:
    def __init__(self):
        self.sample_rate = 250
        self.t_buffer = 300 
        self.thread_data_server = DataServerThread(self.sample_rate, self.t_buffer)
        self.flagstop = False
        self.triggerbox = TriggerBox("COM3")

    def start_data_collection(self):
        notConnect = self.thread_data_server.connect(hostname='127.0.0.1', port=8712)
        if notConnect:
            raise Exception("Can't connect to JellyFish, please check the hostport.")
        else:
            while not self.thread_data_server.isReady():
                time.sleep(0.1)
                continue
            self.thread_data_server.start()
            print("Data collection started.")

    def trigger(self, label):
        code = int(label)  # 直接将传入的类别编号转换为整数
        print(f'Sending trigger for label {label}: {code}')
        self.triggerbox.output_event_data(code)

    def stop_data_collection(self):
        self.flagstop = True
        self.thread_data_server.stop()

    def save_pre_eeg(self, pre_eeg_path):
        original_data_path = os.path.join(pre_eeg_path, f'original\{time.strftime("%Y%m%d-%H%M%S")}.npy')
        preprocess_data_path = os.path.join(pre_eeg_path, f'preprocessed\{time.strftime("%Y%m%d-%H%M%S")}.npy')
        
        # 确保目录存在
        os.makedirs(os.path.dirname(original_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(preprocess_data_path), exist_ok=True)

        data = self.thread_data_server.GetBufferData()
        np.save(original_data_path, data)
        print("Pre-experiment data saved!")

        # 进行数据预处理
        filters = prepare_filters(fs = self.sample_rate, new_fs=250)
        real_time_processing(original_data_path, preprocess_data_path, filters)
        print("Pre-experiment data preprocessed!")

        # 数据 event-based 处理
        create_event_based_npy(original_data_path, preprocess_data_path, pre_eeg_path)
        
    def save_labels(self, labels, path):
        labels_path = os.path.join(path, 'labels.npy')
        # 确保目录存在
        os.makedirs(os.path.dirname(labels_path), exist_ok=True)
        # 保存标签
        np.save(labels_path, labels)
        print("Labels saved!")

    def save_instant_eeg(self, instant_eeg_path):
        data = self.thread_data_server.GetBufferData()
        event_data_list = create_last_event_npy(data, 1)
        event_data = event_data_list[0]  # 取第一个事件数据
        filters = prepare_filters(fs = self.sample_rate, new_fs=250)
        data = real_time_process(event_data, filters)
        np.save(os.path.join(instant_eeg_path, f'{time.strftime("%Y%m%d-%H%M%S")}.npy'), data)
        print("Instant EEG data saved!")
        
    def save_eeg(self, instant_eeg_path, file_name):
        # 确保目录存在
        os.makedirs(instant_eeg_path, exist_ok=True)
        # 保存数据
        data = self.thread_data_server.GetBufferData()
        np.save(os.path.join(instant_eeg_path, f'{file_name}.npy'), data)
        print("Instant EEG data saved!")

    def get_event_data(self):
        data = self.thread_data_server.GetBufferData()
        event_data_list = create_last_event_npy(data, 1)
        return event_data_list
    
    def get_next_sequence(self):
        # 确保不会超出列表范围
        if self.current_sequence * self.num_per_event >= len(self.sequence_indices):
            raise Exception("All sequences have been displayed.")

        # 从打乱的索引列表中获取下一个序列的索引
        sequence_start_index = self.current_sequence * self.num_per_event
        sequence_end_index = sequence_start_index + self.num_per_event
        next_sequence_indices = self.sequence_indices[sequence_start_index:sequence_end_index]

        # 更新当前序列计数
        self.current_sequence += 1

        # 返回选中的图像和标签，即返回 20 个 images_with_labels 元组
        return [(self.images[i], i) for i in next_sequence_indices]

    def reset_sequence(self):
        self.current_sequence = 0

    def set_phase(self, phase):
        self.current_phase = phase

class BaseController:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.running = True
        
    def process_events(self):
        """处理所有排队的事件，提高响应性能"""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            # 将键盘事件传递给视图进行处理
            if hasattr(self.view, 'handle_event'):
                self.view.handle_event(event)
        return self.running
    
    def run(self):
        self.running = True
        clock = pg.time.Clock()
        
        while self.running:
            self.running = self.process_events()
            
            clock.tick(60)  # 控制帧率

    def start_rating(self, instant_image_path):
        print("Start rating")
        self.view.display_text('Ready to start rating')
        
        # 获取所有图片文件
        all_image_files = [f for f in os.listdir(instant_image_path) if f.endswith('.jpg') or f.endswith('.png')]
        
        # 初始化标签列表
        ratings = []
        
        # 显示图片并采集数据
        for image_file in all_image_files:
            image_path = os.path.join(instant_image_path, image_file)
            print(f"显示图片: {image_path}")
            
            image = pg.image.load(image_path)
            self.view.display_image(image)
            start_time = pg.time.get_ticks()
            while pg.time.get_ticks() - start_time < 2000:
                self.process_events()
                pg.time.delay(10)             
            score = self.view.rating()
            if score is None:
                score = 0.5  # 默认评分
            ratings.append(score)
            print(f"评分: {score}")
        return ratings

    def end_experiment(self):
        self.view.display_text('Thank you!')
        time.sleep(3)
        self.model.stop_data_collection()
        pg.quit()
        gc.collect()
        quit()

class EEGController:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.running = True
        self.model.start_data_collection()
        
    def process_events(self):
        """处理所有排队的事件，提高响应性能"""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            # 将键盘事件传递给视图进行处理
            if hasattr(self.view, 'handle_event'):
                self.view.handle_event(event)
        return self.running
    
    def run(self):
        self.running = True
        clock = pg.time.Clock()
        
        while self.running:
            self.running = self.process_events()
            
            clock.tick(60)  # 控制帧率
            
    
    def start_collect_and_rating(self, instant_image_path, instant_eeg_path):
        print("Start rating")
        self.view.display_text('Ready to start rating')
        
        # 获取所有图片文件
        all_image_files = [f for f in os.listdir(instant_image_path) if f.endswith('.jpg') or f.endswith('.png')]
        
        # 初始化标签列表
        ratings = []
        count = 0
        
        # 显示图片并采集数据
        for image_file in all_image_files:
            count+= 1
            image_path = os.path.join(instant_image_path, image_file)
            print(f"显示图片: {image_path}")
            
            image = pg.image.load(image_path)
            self.view.display_image(image)
            self.model.trigger(count)
            start_time = pg.time.get_ticks()
            while pg.time.get_ticks() - start_time < 5000:
                self.process_events()
                pg.time.delay(10)             
            score = self.view.rating()
            if score is None:
                score = 0.5  # 默认评分
            ratings.append(score)
            print(f"评分: {score}")
        self.model.save_eeg(instant_eeg_path, '1')
        return ratings
    
    def rating(self):
        self.view.display_text('Please rate the image')
        time.sleep(1)
        self.view.clear_screen()

    def end_experiment(self):
        self.view.display_text('Thank you!')
        time.sleep(3)
        self.model.stop_data_collection()
        pg.quit()
        gc.collect()
        quit()


class View:
    def __init__(self):
        pg.init()
        
        # 创建固定尺寸的窗口 1920x1080
        screen_width, screen_height = 1920, 1080
        self.screen = pg.display.set_mode((screen_width, screen_height))
        pg.display.set_caption('Closed Loop Experiment')
        self.font = pg.font.Font(None, 40)
        
        # 存储窗口信息
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # 对于输入相关的状态变量
        self.input_active = False
        self.input_text = ""
        self.input_rect = pg.Rect(screen_width//2 - 100, screen_height//2, 200, 50)
        self.input_result = None
        
        print(f"已创建 {screen_width}x{screen_height} 窗口")
        
    def handle_event(self, event):
        """处理事件，提高键盘响应性能"""
        if not self.input_active:
            return
            
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_RETURN:
                try:
                    score = float(self.input_text)
                    if 0 <= score <= 1:
                        self.input_result = score
                        self.input_active = False
                except ValueError:
                    pass
            elif event.key == pg.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            elif event.key == pg.K_ESCAPE:
                self.input_active = False
                self.input_result = None
            elif event.unicode in '0123456789.' and len(self.input_text) < 4:
                # 确保只有一个小数点和小数点后最多两位
                if event.unicode == '.' and '.' in self.input_text:
                    return
                if '.' in self.input_text and len(self.input_text.split('.')[1]) >= 2 and event.unicode != '.':
                    return
                self.input_text += event.unicode
                
            # 实时更新显示
            self.update_rating_display()
        
    def display_text(self, text):
        self.screen.fill((0, 0, 0))
        text_surface = self.font.render(text, True, (255, 255, 255))
        self.screen.blit(text_surface, (self.screen.get_width() // 2 - text_surface.get_width() // 2,
                                        self.screen.get_height() // 2 - text_surface.get_height() // 2))
        pg.display.flip()     

    def display_fixation(self):
        self.screen.fill((0, 0, 0))  # 清屏
        # 绘制红色圆
        pg.draw.circle(self.screen, (200, 0, 0), (400,300), 10, 0)
        # 绘制黑色十字
        pg.draw.line(self.screen, (0, 0, 0), (425, 300), (375, 300), 3)
        pg.draw.line(self.screen, (0, 0, 0), (400, 325), (400, 275), 3)
        pg.display.flip()

    def display_image(self, image):
        # 获取当前屏幕分辨率
        screen_width, screen_height = self.screen.get_size()
        
        # 获取图片的原始尺寸
        img_width, img_height = image.get_size()
        
        # 计算宽高比
        width_ratio = screen_width / img_width
        height_ratio = screen_height / img_height
        
        # 选择较小的比例以确保图片完全显示在屏幕内
        scale_ratio = min(width_ratio, height_ratio)
        
        # 计算缩放后的尺寸
        new_width = int(img_width * scale_ratio)
        new_height = int(img_height * scale_ratio)
        
        # 缩放图片
        scaled_image = pg.transform.scale(image, (new_width, new_height))
        
        # 计算居中位置
        x_pos = (screen_width - new_width) // 2
        y_pos = (screen_height - new_height) // 2
        
        # 先填充黑色背景
        self.screen.fill((0, 0, 0))
        
        # 在居中位置绘制图片
        self.screen.blit(scaled_image, (x_pos, y_pos))
        
        # 更新屏幕显示
        pg.display.flip()

    def clear_screen(self):
        self.screen.fill((0, 0, 0))
        pg.display.flip()

    def display_multiline_text(self, text, position, font_size, line_spacing):
        font = pg.font.Font(self.font_path, font_size)
        lines = text.splitlines()  # 分割文本为多行
        x, y = position

        for line in lines:
            line_surface = font.render(line, True, (255, 255, 255))
            self.screen.blit(line_surface, (x, y))
            y += line_surface.get_height() + line_spacing  # 更新y坐标，为下一行做准备

        pg.display.flip()  # 更新屏幕显示
        
    def handle_event(self, event):
        """处理事件，提高键盘响应性能"""
        if not self.input_active:
            return
            
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_RETURN:
                try:
                    score = float(self.input_text)
                    if 0 <= score <= 1:
                        self.input_result = score
                        self.input_active = False
                except ValueError:
                    pass
            elif event.key == pg.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            elif event.key == pg.K_ESCAPE:
                self.input_active = False
                self.input_result = None
            elif event.unicode in '0123456789.' and len(self.input_text) < 4:
                # 确保只有一个小数点和小数点后最多两位
                if event.unicode == '.' and '.' in self.input_text:
                    return
                if '.' in self.input_text and len(self.input_text.split('.')[1]) >= 2 and event.unicode != '.':
                    return
                self.input_text += event.unicode
                
            # 实时更新显示
            self.update_rating_display()
            
    def update_rating_display(self):
        """更新评分界面显示"""
        if not self.input_active:
            return
            
        # 清屏
        self.screen.fill((0, 0, 0))
        
        # 绘制说明文本
        instruction_text = self.font.render("Input the rating of the image:(0.00-1.00)", True, (255, 255, 255))
        self.screen.blit(instruction_text, (self.screen_width//2 - instruction_text.get_width()//2, 
                                            self.screen_height//2 - 80))
        
        # 绘制输入框
        pg.draw.rect(self.screen, (255, 255, 255), self.input_rect, 2)
        
        # 显示当前输入的文本
        text_surface = self.font.render(self.input_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (self.input_rect.x + 10, self.input_rect.y + 10))
        
        # 显示用法提示
        hint_text = self.font.render("Enter or ESC", True, (200, 200, 200))
        self.screen.blit(hint_text, (self.screen_width//2 - hint_text.get_width()//2, 
                                        self.screen_height//2 + 80))
        
        # 更新显示
        pg.display.flip()
        
    def rating(self):
        """显示评分界面，让用户输入0-1之间的两位小数"""
        print("正在显示打分")
        
        self.input_active = True
        self.input_text = ""
        self.input_result = None
        
        # 初始显示
        self.update_rating_display()
        
        # 等待用户完成输入
        while self.input_active:
            pg.time.delay(10)  # 短暂延迟，降低CPU使用率
            # 事件处理由Controller中的process_events调用handle_event函数完成
        
        return self.input_result