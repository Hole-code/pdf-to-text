# Используем базовый образ с поддержкой CUDA
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# Устанавливаем переменную окружения для избежания интерактивных запросов при установке
ENV DEBIAN_FRONTEND=noninteractive

# Устанавливаем необходимые пакеты
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    tesseract-ocr \
    tesseract-ocr-rus \
    poppler-utils \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы с кодом и зависимостями
COPY requirements.txt .
COPY main.py .

# Обновляем pip и устанавливаем базовые зависимости
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Устанавливаем packaging отдельно
RUN pip3 install --no-cache-dir packaging

# Устанавливаем остальные зависимости
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install accelerate

RUN pip3 install flash-attn --no-build-isolation

# Открываем порт для FastAPI
EXPOSE 8000

# Запускаем FastAPI приложение при старте контейнера
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]