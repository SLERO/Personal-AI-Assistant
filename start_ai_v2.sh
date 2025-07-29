#!/bin/bash
# Проверка и освобождение порта 7860
PORT=7861
if sudo netstat -tuln | grep $PORT; then
    echo "Порт $PORT занят, убиваю процесс..."
    PID=$(sudo netstat -tulnp | grep $PORT | awk '{print $7}' | cut -d/ -f1)
    sudo kill -9 $PID
fi
PROJECT_DIR=/home/dragon/ai_assistant
VENV_PATCH=/home/dragon/ai_assistant
cd $PROJECT_DIR
source $VENV_PATCH/bin/activate
python3 /home/dragon/shared/saiga_chat.py
