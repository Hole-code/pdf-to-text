# PDF Title Extractor

PDF Title Extractor - это инструмент, который позволяет извлекать заголовки из PDF документов. Проект использует технологии NVIDIA для ускоренной обработки данных с помощью GPU.

## Начало работы

Эти инструкции помогут вам запустить проект на вашем локальном компьютере.

### Предварительные требования

Перед тем, как начать, убедитесь, что у вас установлены следующие компоненты:

- Docker
- GPU от NVIDIA

### Установка

1. Добавьте GPG ключ и репозиторий NVIDIA контейнеров:

    ```sh
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    ```

2. Активируйте экспериментальные репозитории:

    ```sh
    sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
    ```

3. Обновите список пакетов:

    ```sh
    sudo apt-get update
    ```

4. Установите NVIDIA Container Toolkit:

    ```sh
    sudo apt-get install -y nvidia-container-toolkit
    ```

5. Постройте Docker образ:

    ```sh
    docker build -t project .
    ```

6. Запустите контейнер с использованием GPU:

    ```sh
    docker run --gpus all -p 8000:8000 project
    ```

## Использование

После запуска контейнера, сервис будет доступен по адресу `http://localhost:8000`. Вы можете отправить PDF файл на этот адрес, чтобы получить заголовок документа.

### Тестирование API

Для тестирования API вы можете использовать следующую команду curl:

```sh
curl -X POST http://ip:8000/parse -H 'Content-Type: multipart/form-data' -F 'file=@sa.pdf'
```

Эта команда отправляет PDF файл sa.pdf на сервер для извлечения заголовка.

## Требования к оборудованию
Проект был протестирован на следующей конфигурации оборудования:

Видеокарта: Tesla A100 с 80 гигабайтами видеопамяти
Оперативная память: 64 гигабайта
Процессор: 16 ядер
Для работы модели необходимо:

Более 50 гигабайт видеопамяти
Архитектура GPU: Ampere
