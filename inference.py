import os
import tensorflow as tf
from models.unet_model import create_model
from utils.video_processor import VideoProcessor
from config import Config

def load_model(config):
    """Загрузка модели с весами"""
    model = create_model(config)
    
    if os.path.exists(config.MODEL_WEIGHTS):
        print("Загрузка весов модели...")
        model.load_weights(config.MODEL_WEIGHTS)
    else:
        print("Предупреждение: веса модели не найдены. Используется новая модель.")
        # Создание директории для весов
        os.makedirs('weights', exist_ok=True)
    
    return model

def main():
    # Загрузка конфигурации
    config = Config()
    
    # Загрузка модели
    print("Инициализация модели...")
    model = load_model(config)
    
    # Создание процессора видео
    processor = VideoProcessor(model, config)
    
    # Обработка видео
    print("Начало обработки видео...")
    processor.process_video(config.VIDEO_PATH, config.OUTPUT_PATH)
    
    print(f"Результат сохранен в: {config.OUTPUT_PATH}")

if __name__ == "__main__":
    main()