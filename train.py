import tensorflow as tf
import numpy as np
import os
from models.unet_model import create_model
from config import Config

def create_dummy_data(config):
    """Создание dummy данных для демонстрации"""
    # В реальном проекте здесь должна быть загрузка вашего датасета
    x_train = np.random.random((10, config.INPUT_HEIGHT, config.INPUT_WIDTH, 3))
    y_train = tf.keras.utils.to_categorical(
        np.random.randint(0, config.NUM_CLASSES, (10, config.INPUT_HEIGHT, config.INPUT_WIDTH)),
        num_classes=config.NUM_CLASSES
    )
    return x_train, y_train

def train_model():
    config = Config()
    
    # Создание модели
    model = create_model(config)
    model.summary()
    
    # Создание данных
    x_train, y_train = create_dummy_data(config)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            config.MODEL_WEIGHTS,
            save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3),
        tf.keras.callbacks.EarlyStopping(patience=5)
    ]
    
    # Обучение
    history = model.fit(
        x_train, y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_split=0.2,
        callbacks=callbacks
    )
    
    print("Обучение завершено!")

if __name__ == "__main__":
    train_model()