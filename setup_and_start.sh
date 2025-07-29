#!/bin/bash

# Папки
PROJECT_DIR=/home/dragon/ai_assistant
SHARED_DIR=/home/dragon/shared

# 1. Пересоздаём venv (если нужно)
rm -rf $PROJECT_DIR  # Удаляем старый (если сломан)
python3 -m venv $PROJECT_DIR

# 2. Активируем и устанавливаем зависимости
cd $PROJECT_DIR
source bin/activate
pip install -r $SHARED_DIR/requirements.txt  # Ваш файл из shared (или укажите путь к portable/full)
pip install llama-cpp-python gradio pydub pyaudio  # Добавляем ключевые, на случай

# 3. Переходим в shared и запускаем
cd $SHARED_DIR
python3 saiga_chat.py