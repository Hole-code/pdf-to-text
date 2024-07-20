import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import PyPDF2
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
from deep_translator import GoogleTranslator
import time
from transformers import AutoModel, AutoTokenizer
import torch

import math
import numpy as np
import torchvision.transforms as T
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
import re

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)



def parse_text(text):
    # Регулярные выражения для поиска нужной информации
    date_pattern = re.compile(r"Date: (.+)")
    author_pattern = re.compile(r"Author: (.+)")
    title_pattern = re.compile(r"Text Title: (.+)")

    # Поиск совпадений
    date_match = date_pattern.search(text)
    author_match = author_pattern.search(text)
    title_match = title_pattern.search(text)

    # Извлечение найденных значений
    date = date_match.group(1) if date_match else None
    author = author_match.group(1) if author_match else None
    title = title_match.group(1) if title_match else None

    return {
        'date': date,
        'author': author,
        'title': title
    }


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {'InternVL2-8B': 32, 'InternVL2-26B': 48,
                  'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


path = 'OpenGVLab/InternVL2-26B'
# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
# Otherwise, you need to set device_map to use multiple GPUs for inference.
# device_map = split_model('InternVL2-26B')
# print(device_map)
# model = AutoModel.from_pretrained(
#     path,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
#     device_map=device_map).eval()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

generation_config = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)

app = FastAPI()

@app.post("/parse")
async def parse_pdf(file: UploadFile = File(...)):
    if file.filename.split(".")[-1].lower() != "pdf":
        return JSONResponse(status_code=400, content={"error": "Uploaded file must be a PDF"})
    
    pdf_content = await file.read()
    
    # Конвертация PDF в изображения
    images = convert_from_bytes(pdf_content)
    
    # OCR для каждого изображения
    full_text = ""
    for image in images:
        text = pytesseract.image_to_string(image,lang='rus')
        full_text += text + "\n\n"

    # Перевод текста с русского на английский
    translator = GoogleTranslator(source='ru', target='en')
    translated_text = translator.translate(full_text)
    
    # Извлечение метаданных из PDF
    pdf_file = io.BytesIO(pdf_content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    s = """
    Find it in the text
    Date of publication, author of the text, title of the text 
    Write them in the format
    Date:
    Author:
    Text Title:

    """
    question = s+translated_text
    response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
    print(f'User: {question}')
    print(f'Assistant: {response}')
    parsed_info = parse_text(response)

    translator = GoogleTranslator(source='en', target='ru')
    title = translator.translate(parsed_info["title"])
    authors = translator.translate(parsed_info["author"])
    publication_date = translator.translate(parsed_info["date"])
    
    creation_date = pdf_reader.metadata.get('/CreationDate', '')
    if creation_date.startswith('D:'):
        publication_date = creation_date[2:6]  # Извлекаем только год
    else:
        publication_date = 'Unknown'

    return JSONResponse(content={
        "fields": {
            "title": title,
            "authors": authors,
            "publication_date": publication_date,
            "response": response
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)