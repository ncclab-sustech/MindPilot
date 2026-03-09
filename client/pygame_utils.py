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
        code = int(label)  # Directly convert the passed-in category number to an integer
        print(f'Sending trigger for label {label}: {code}')
        self.triggerbox.output_event_data(code)

    def stop_data_collection(self):
        self.flagstop = True
        self.thread_data_server.stop()

    def save_pre_eeg(self, pre_eeg_path):
        original_data_path = os.path.join(pre_eeg_path, f'original\{time.strftime("%Y%m%d-%H%M%S")}.npy')
        preprocess_data_path = os.path.join(pre_eeg_path, f'preprocessed\{time.strftime("%Y%m%d-%H%M%S")}.npy')
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(original_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(preprocess_data_path), exist_ok=True)

        data = self.thread_data_server.GetBufferData()
        np.save(original_data_path, data)
        print("Pre-experiment data saved!")

        # Perform data preprocessing
        filters = prepare_filters(fs = self.sample_rate, new_fs=250)
        real_time_processing(original_data_path, preprocess_data_path, filters)
        print("Pre-experiment data preprocessed!")

        # Event-based data processing
        create_event_based_npy(original_data_path, preprocess_data_path, pre_eeg_path)
        
    def save_labels(self, labels, path):
        labels_path = os.path.join(path, 'labels.npy')
        # Ensure the directory exists
        os.makedirs(os.path.dirname(labels_path), exist_ok=True)
        # Save labels
        np.save(labels_path, labels)
        print("Labels saved!")

    def save_instant_eeg(self, instant_eeg_path):
        data = self.thread_data_server.GetBufferData()
        event_data_list = create_last_event_npy(data, 1)
        event_data = event_data_list[0]  # Take the first event data
        filters = prepare_filters(fs = self.sample_rate, new_fs=250)
        data = real_time_process(event_data, filters)
        np.save(os.path.join(instant_eeg_path, f'{time.strftime("%Y%m%d-%H%M%S")}.npy'), data)
        print("Instant EEG data saved!")
        
    def save_eeg(self, instant_eeg_path, file_name):
        # Ensure the directory exists
        os.makedirs(instant_eeg_path, exist_ok=True)
        # Save data
        data = self.thread_data_server.GetBufferData()
        np.save(os.path.join(instant_eeg_path, f'{file_name}.npy'), data)
        print("Instant EEG data saved!")

    def get_event_data(self):
        data = self.thread_data_server.GetBufferData()
        event_data_list = create_last_event_npy(data, 1)
        return event_data_list
    
    def get_next_sequence(self):
        # Ensure we don't exceed the list bounds
        if self.current_sequence * self.num_per_event >= len(self.sequence_indices):
            raise Exception("All sequences have been displayed.")

        # Get the next sequence indices from the shuffled index list
        sequence_start_index = self.current_sequence * self.num_per_event
        sequence_end_index = sequence_start_index + self.num_per_event
        next_sequence_indices = self.sequence_indices[sequence_start_index:sequence_end_index]

        # Update the current sequence counter
        self.current_sequence += 1

        # Return the selected images and labels, i.e. 20 (image, label) tuples
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
        """Process all queued events to improve responsiveness."""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            # Pass keyboard events to the view for handling
            if hasattr(self.view, 'handle_event'):
                self.view.handle_event(event)
        return self.running
    
    def run(self):
        self.running = True
        clock = pg.time.Clock()
        
        while self.running:
            self.running = self.process_events()
            
            clock.tick(60)  # Control frame rate

    def start_rating(self, instant_image_path):
        print("Start rating")
        self.view.display_text('Ready to start rating')
        
        # Get all image files
        all_image_files = [f for f in os.listdir(instant_image_path) if f.endswith('.jpg') or f.endswith('.png')]
        
        # Initialize ratings list
        ratings = []
        
        # Display images and collect data
        for image_file in all_image_files:
            image_path = os.path.join(instant_image_path, image_file)
            print(f"Displaying image: {image_path}")
            
            image = pg.image.load(image_path)
            self.view.display_image(image)
            start_time = pg.time.get_ticks()
            while pg.time.get_ticks() - start_time < 2000:
                self.process_events()
                pg.time.delay(10)             
            score = self.view.rating()
            if score is None:
                score = 0.5  # Default rating
            ratings.append(score)
            print(f"Rating: {score}")
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
        """Process all queued events to improve responsiveness."""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            # Pass keyboard events to the view for handling
            if hasattr(self.view, 'handle_event'):
                self.view.handle_event(event)
        return self.running
    
    def run(self):
        self.running = True
        clock = pg.time.Clock()
        
        while self.running:
            self.running = self.process_events()
            
            clock.tick(60)  # Control frame rate
            
    
    def start_collect_and_rating(self, instant_image_path, instant_eeg_path):
        print("Start rating")
        self.view.display_text('Ready to start rating')
        
        # Get all image files
        all_image_files = [f for f in os.listdir(instant_image_path) if f.endswith('.jpg') or f.endswith('.png')]
        
        # Initialize ratings list
        ratings = []
        count = 0
        
        # Display images and collect data
        for image_file in all_image_files:
            count+= 1
            image_path = os.path.join(instant_image_path, image_file)
            print(f"Displaying image: {image_path}")
            
            image = pg.image.load(image_path)
            self.view.display_image(image)
            self.model.trigger(count)
            start_time = pg.time.get_ticks()
            while pg.time.get_ticks() - start_time < 5000:
                self.process_events()
                pg.time.delay(10)             
            score = self.view.rating()
            if score is None:
                score = 0.5  # Default rating
            ratings.append(score)
            print(f"Rating: {score}")
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
        
        # Create a fixed-size window 1920x1080
        screen_width, screen_height = 1920, 1080
        self.screen = pg.display.set_mode((screen_width, screen_height))
        pg.display.set_caption('Closed Loop Experiment')
        self.font = pg.font.Font(None, 40)
        
        # Store window information
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # State variables for input handling
        self.input_active = False
        self.input_text = ""
        self.input_rect = pg.Rect(screen_width//2 - 100, screen_height//2, 200, 50)
        self.input_result = None
        
        print(f"Created {screen_width}x{screen_height} window")
        
    def handle_event(self, event):
        """Handle events to improve keyboard responsiveness."""
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
                # Ensure only one decimal point and at most two decimal places
                if event.unicode == '.' and '.' in self.input_text:
                    return
                if '.' in self.input_text and len(self.input_text.split('.')[1]) >= 2 and event.unicode != '.':
                    return
                self.input_text += event.unicode
                
            # Update display in real time
            self.update_rating_display()
        
    def display_text(self, text):
        self.screen.fill((0, 0, 0))
        text_surface = self.font.render(text, True, (255, 255, 255))
        self.screen.blit(text_surface, (self.screen.get_width() // 2 - text_surface.get_width() // 2,
                                        self.screen.get_height() // 2 - text_surface.get_height() // 2))
        pg.display.flip()     

    def display_fixation(self):
        self.screen.fill((0, 0, 0))  # Clear screen
        # Draw red circle
        pg.draw.circle(self.screen, (200, 0, 0), (400,300), 10, 0)
        # Draw black cross
        pg.draw.line(self.screen, (0, 0, 0), (425, 300), (375, 300), 3)
        pg.draw.line(self.screen, (0, 0, 0), (400, 325), (400, 275), 3)
        pg.display.flip()

    def display_image(self, image):
        # Get current screen resolution
        screen_width, screen_height = self.screen.get_size()
        
        # Get the original image dimensions
        img_width, img_height = image.get_size()
        
        # Calculate aspect ratios
        width_ratio = screen_width / img_width
        height_ratio = screen_height / img_height
        
        # Choose the smaller ratio to ensure the image fits entirely on screen
        scale_ratio = min(width_ratio, height_ratio)
        
        # Calculate scaled dimensions
        new_width = int(img_width * scale_ratio)
        new_height = int(img_height * scale_ratio)
        
        # Scale the image
        scaled_image = pg.transform.scale(image, (new_width, new_height))
        
        # Calculate centered position
        x_pos = (screen_width - new_width) // 2
        y_pos = (screen_height - new_height) // 2
        
        # Fill with black background first
        self.screen.fill((0, 0, 0))
        
        # Draw the image at the centered position
        self.screen.blit(scaled_image, (x_pos, y_pos))
        
        # Update screen display
        pg.display.flip()

    def clear_screen(self):
        self.screen.fill((0, 0, 0))
        pg.display.flip()

    def display_multiline_text(self, text, position, font_size, line_spacing):
        font = pg.font.Font(self.font_path, font_size)
        lines = text.splitlines()  # Split text into multiple lines
        x, y = position

        for line in lines:
            line_surface = font.render(line, True, (255, 255, 255))
            self.screen.blit(line_surface, (x, y))
            y += line_surface.get_height() + line_spacing  # Update y coordinate for the next line

        pg.display.flip()  # Update screen display
        
    def handle_event(self, event):
        """Handle events to improve keyboard responsiveness."""
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
                # Ensure only one decimal point and at most two decimal places
                if event.unicode == '.' and '.' in self.input_text:
                    return
                if '.' in self.input_text and len(self.input_text.split('.')[1]) >= 2 and event.unicode != '.':
                    return
                self.input_text += event.unicode
                
            # Update display in real time
            self.update_rating_display()
            
    def update_rating_display(self):
        """Update the rating interface display."""
        if not self.input_active:
            return
            
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Draw instruction text
        instruction_text = self.font.render("Input the rating of the image:(0.00-1.00)", True, (255, 255, 255))
        self.screen.blit(instruction_text, (self.screen_width//2 - instruction_text.get_width()//2, 
                                            self.screen_height//2 - 80))
        
        # Draw input box
        pg.draw.rect(self.screen, (255, 255, 255), self.input_rect, 2)
        
        # Display current input text
        text_surface = self.font.render(self.input_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (self.input_rect.x + 10, self.input_rect.y + 10))
        
        # Display usage hint
        hint_text = self.font.render("Enter or ESC", True, (200, 200, 200))
        self.screen.blit(hint_text, (self.screen_width//2 - hint_text.get_width()//2, 
                                        self.screen_height//2 + 80))
        
        # Update display
        pg.display.flip()
        
    def rating(self):
        """Display the rating interface for the user to input a decimal between 0 and 1."""
        print("Displaying rating interface")
        
        self.input_active = True
        self.input_text = ""
        self.input_result = None
        
        # Initial display
        self.update_rating_display()
        
        # Wait for user to complete input
        while self.input_active:
            pg.time.delay(10)  # Brief delay to reduce CPU usage
            # Event handling is done by Controller's process_events calling handle_event
        
        return self.input_result