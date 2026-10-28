import os

class Config:
    # Размеры изображения
    IMAGE_HEIGHT = 1142
    IMAGE_WIDTH = 1888
    # Для обучения можно использовать меньший размер
    INPUT_HEIGHT = 512
    INPUT_WIDTH = 512
    
    # Классы для сегментации
    CLASSES = ['background', 'person', 'car']
    NUM_CLASSES = len(CLASSES)
    
    # Пути
    MODEL_WEIGHTS = 'weights/unet_weights.h5'
    VIDEO_PATH = 'data/input/input_video.mp4'
    OUTPUT_PATH = 'output_video.mp4'
    
    # Параметры модели
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    
config = Config()