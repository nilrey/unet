import cv2
import numpy as np
import tensorflow as tf
from config import Config

class VideoProcessor:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def preprocess_frame(self, frame):
        """Подготовка кадра для модели"""
        # Изменение размера
        frame_resized = cv2.resize(frame, (self.config.INPUT_WIDTH, self.config.INPUT_HEIGHT))
        # Нормализация
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        # Добавление batch dimension
        frame_batch = np.expand_dims(frame_normalized, axis=0)
        return frame_batch
    
    def postprocess_mask(self, mask, original_shape):
        """Постобработка маски"""
        # Получение класса с максимальной вероятностью
        class_mask = np.argmax(mask, axis=-1)
        # Изменение размера к оригинальному
        class_mask = class_mask[0]  # Убираем batch dimension
        mask_resized = cv2.resize(class_mask, (original_shape[1], original_shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
        return mask_resized
    
    def process_video(self, input_path, output_path):
        """Обработка всего видео"""
        cap = cv2.VideoCapture(input_path)
        
        # Получение параметров видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Создание VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Обработка видео: {total_frames} кадров")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Обработка кадра
            processed_frame = self.process_frame(frame)
            out.write(processed_frame)
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Обработано кадров: {frame_count}/{total_frames}")
        
        cap.release()
        out.release()
        print("Обработка завершена!")
    
    def process_frame(self, frame):
        """Обработка одного кадра"""
        original_shape = frame.shape
        
        # Предобработка
        processed_frame = self.preprocess_frame(frame)
        
        # Предсказание
        mask = self.model.predict(processed_frame, verbose=0)
        
        # Постобработка маски
        segmentation_mask = self.postprocess_mask(mask, original_shape)
        
        # Визуализация результата
        result_frame = self.visualize_segmentation(frame, segmentation_mask)
        
        return result_frame
    
    def visualize_segmentation(self, frame, mask):
        """Визуализация сегментации на кадре"""
        # Цвета для классов
        colors = {
            0: [0, 0, 0],      # фон - черный
            1: [0, 255, 0],    # люди - зеленый
            2: [255, 0, 0]     # машины - красный
        }
        
        # Создание цветной маски
        colored_mask = np.zeros_like(frame)
        for class_id, color in colors.items():
            colored_mask[mask == class_id] = color
        
        # Наложение маски на оригинальный кадр
        alpha = 0.5
        result = cv2.addWeighted(frame, 1 - alpha, colored_mask, alpha, 0)
        
        # Добавление легенды
        legend_height = 100
        legend = np.zeros((legend_height, frame.shape[1], 3), dtype=np.uint8)
        
        y_offset = 20
        for i, (class_name, color) in enumerate(zip(self.config.CLASSES, colors.values())):
            if class_name == 'background':
                continue
            cv2.rectangle(legend, (20, y_offset + i*30), (50, y_offset + i*30 + 20), color, -1)
            cv2.putText(legend, class_name, (60, y_offset + i*30 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Объединение кадра и легенды
        result_with_legend = np.vstack([result, legend])
        
        return result_with_legend