import torch
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from functools import partial
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from enum import Enum
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import aiohttp
import os
from PIL import Image
import io
from aiogram.utils import markdown as md
import base64
import telebot
from ultralytics import YOLO
import torchvision.transforms as transforms
import torchvision.models as models
import itertools

import warnings
warnings.filterwarnings("ignore")

TOKEN = 'your token'
bot = Bot(token=TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot=bot, storage=storage)

# Для текста
model_name = "cointegrated/rubert-tiny2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
df = pd.read_csv('csv/tsum_cat.csv')
categories = ['up', 'bottom', 'full', 'shoes', 'acc', 'bag']
length = 256
with open("embeddings_256.pkl", "rb") as f:
    embeddings_text = pickle.load(f)

# Для фото
model_weights = 'fashion-pretrained-best.pt'
detection_model = YOLO(model_weights)
embeddings_by_category = {}
yolo_category = ['sunglass', 'hat', 'jacket', 'shirt',
                 'pants', 'shorts', 'skirt', 'dress', 'bag', 'shoe']
folder_path = "embeddings"

for category in yolo_category:
    file_path = os.path.join(folder_path, f"{category}.pkl")

    if os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            embeddings = pickle.load(f)
        embeddings_by_category[category] = embeddings
    else:
        embeddings_by_category[category] = []

embeddings_sunglass = embeddings_by_category['sunglass']
embeddings_hat = embeddings_by_category['hat']
embeddings_jacket = embeddings_by_category['jacket']
embeddings_shirt = embeddings_by_category['shirt']
embeddings_pants = embeddings_by_category['pants']
embeddings_shorts = embeddings_by_category['shorts']
embeddings_skirt = embeddings_by_category['skirt']
embeddings_dress = embeddings_by_category['dress']
embeddings_bag = embeddings_by_category['bag']
embeddings_shoe = embeddings_by_category['shoe']



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = models.resnet50(pretrained=True).to(device)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()



def generate_recommendations(query, embeddings):
    top_elements_per_query = defaultdict(lambda: defaultdict(list))

    for category in categories:
        query_tokens = tokenizer.encode_plus(
            query,
            add_special_tokens=True,
            max_length=length,
            pad_to_max_length=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            query_outputs = model(**query_tokens)
            query_hidden_states = query_outputs.hidden_states
            query_last_hidden_state = query_hidden_states[-1][:, 0, :]
            query_embedding = query_last_hidden_state.squeeze()
            cosine_similarities = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                torch.stack(embeddings)
            )

            cosine_similarities = cosine_similarities.numpy()

            indices = np.argsort(cosine_similarities)[::-1]

            elements = []
            for i in indices:
                try:
                    if df['category'][i] == category:
                        url = df['page_url'][i]
                        probability = cosine_similarities[i]
                        elements.append((url, probability))
                        if len(elements) == 1:
                            break
                except KeyError:
                    continue
            top_elements_per_query[query][category] = elements

    temp_basket = []
    category_probabilities = defaultdict(list)

    for query, elements_per_category in top_elements_per_query.items():
        for category, elements in elements_per_category.items():
            for url, probability in elements:
                try:
                    row = df.loc[df['page_url'] == url].iloc[0].tolist()
                    row.append(probability)
                    temp_basket.append(row)
                    category_probabilities[category].append(probability)
                except KeyError:
                    continue

    columns = ['page_url', 'image_url', 'image_url_s', 'image_url_t', 'name', 'cat_1', 'cat_2', 'cat_3',
               'brand', 'price', 'color', 'description', 'image_url_name', 'image_url_s_name',
               'image_url_t_name', 'category', 'yolo_category','probability']
    temp_basket = pd.DataFrame(temp_basket, columns=columns)
    category_probabilities = {category: np.array(probabilities) for category, probabilities in
                              category_probabilities.items()}
    if np.mean(category_probabilities['full']) > np.mean(
            (category_probabilities['up'] + category_probabilities['bottom']) / 2):
        temp_basket = temp_basket[temp_basket['category'] != 'up']
        temp_basket = temp_basket[temp_basket['category'] != 'bottom']
    else:
        temp_basket = temp_basket[temp_basket['category'] != 'full']

    return temp_basket


def get_clothes(img_path):
    cloth_info = []
    cloth_imgs = []

    image = Image.open(img_path)

    items_info = detection_model.predict(source=image, conf=0.5)

    for i, box in enumerate(items_info[0].boxes):
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]

        class_id = round(box.cls[0].item())
        label = yolo_category[class_id]
        conf = round(box.conf[0].item(), 2)

        cloth_img = image.crop(tuple(cords))
        cloth_imgs.append(cloth_img)

        cloth_info.append(label)

    return cloth_imgs, cloth_info


preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    output_tensor = preprocess(image).unsqueeze(0).to(device)
    return output_tensor


def get_embedding(image_path):
    input_var = preprocess_image(image_path)
    with torch.no_grad():
        output = resnet(input_var)
        output = output.view(1, 2048)   
        embedding = torch.nn.functional.normalize(output)
    return embedding


def find_similar_images(query_image_path, label, top_k=1):
    with open(f'txt_for_embeddings/{label}.txt', "r") as file:
        folder_list = file.readlines()
    folder_list = [line.strip() for line in folder_list]

    if label == 'sunglass':
        embeddings = embeddings_sunglass
    elif label == 'hat':
        embeddings = embeddings_hat
    elif label == 'jacket':
        embeddings = embeddings_jacket
    elif label == 'shirt':
        embeddings = embeddings_shirt
    elif label == 'pants':
        embeddings = embeddings_pants
    elif label == 'shorts':
        embeddings = embeddings_shorts
    elif label == 'skirt':
        embeddings = embeddings_skirt
    elif label == 'dress':
        embeddings = embeddings_dress
    elif label == 'bag':
        embeddings = embeddings_bag
    elif label == 'shoe':
        embeddings = embeddings_shoe
    else:
        embeddings = []

    query_embedding = get_embedding(query_image_path)

    distances = []
    for embedding in embeddings:
        distance = torch.nn.functional.cosine_similarity(embedding, query_embedding).item()
        distances.append(distance)

    indices = np.argsort(distances)[::-1][:top_k]
    similar_images = [folder_list[i] for i in indices]
    similar_distances = [distances[i] for i in indices]
    return similar_images, similar_distances


class StyleChoice(StatesGroup):
    process_style = State()
    process_method = State()
    process_photo = State()


async def fetch_image(image_url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status == 200:
                    image_bytes = await response.read()
                    return image_bytes
                else:
                    return None
    except aiohttp.ClientError:
        return None


async def download_image(image_url):
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as response:
            if response.status == 200:
                image_bytes = await response.read()
                return image_bytes
            else:
                return None


async def save_image(image_bytes, file_name):
    with open(file_name, 'wb') as file:
        file.write(image_bytes)


@dp.message_handler(commands=['start'])
async def start(message: types.Message, state: FSMContext):
    await message.reply("Привет! Я твой Personal Shopper. Давай подберем образы. Выбери как бы ты хотела их подобрать?",
                        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                            [InlineKeyboardButton(text="По фотографии", callback_data="photo")],
                            [InlineKeyboardButton(text="По тексту", callback_data="text")],
                        ]))
    await StyleChoice.process_style.set()


@dp.callback_query_handler(state=StyleChoice.process_style)
async def process_style_callback(callback_query: types.CallbackQuery, state: FSMContext):
    method = callback_query.data

    if method == "photo":
        await bot.send_message(callback_query.from_user.id, "Загрузите фото желаемого образа")
        await StyleChoice.process_photo.set()
    elif method == "text":
        await bot.send_message(callback_query.from_user.id, "Какой образ вы хотели бы подобрать?")
        await StyleChoice.process_method.set()

    await callback_query.answer()


@dp.message_handler(state=StyleChoice.process_method)
async def process_method(message: types.Message, state: FSMContext):
    style = message.text.strip()

    await state.update_data(outfit=message.text.strip())  # Сохраняем ответ пользователя в переменной outfit

    query = style

    top_elements_per_category_df = generate_recommendations(query, embeddings_text)

    response = "Рекомендации:\n\n"
    images = []

    for _, row in top_elements_per_category_df.iterrows():
        name = row['name']
        brand = row['brand']
        price = row['price']
        page_url = row['page_url']
        response += f"<a href='{page_url}'>{name}</a>\n{brand}\n{price}\n\n"
        image_url = row['image_url']
        image_data = await fetch_image(image_url)

        # Append the image bytes to the list
        images.append(image_data)

    media_group = []
    for image_bytes in images:
        image_file = io.BytesIO(image_bytes)
        media_group.append(types.InputMediaPhoto(image_file))
    await bot.send_media_group(message.chat.id, media=media_group)

    images.clear()
    top_elements_per_category_df = pd.DataFrame()

    # Reset the state to restart the process
    await state.reset_state()

    # Reply with a new message
    await message.reply(response, parse_mode='HTML', disable_web_page_preview=True)
    await message.reply("Выбери как бы ты хотела подобрать образ:",
                        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                            [InlineKeyboardButton(text="По фотографии", callback_data="photo")],
                            [InlineKeyboardButton(text="По тексту", callback_data="text")],
                        ]))
    await StyleChoice.process_style.set()


@dp.message_handler(content_types=types.ContentType.PHOTO, state=StyleChoice.process_photo)
async def process_photo(message: types.Message, state: FSMContext):
    photo = message.photo[-1]
    photo_id = photo.file_id
    photo_path = await bot.get_file(photo_id)
    
    # Загрузка фотографии и сохранение ее на диске
    folder_temporary = 'temporary_request'
    if not os.path.exists(folder_temporary):
        os.makedirs(folder_temporary)
    photo_file_path = f"{folder_temporary}/photo.jpg"  # Путь к файлу, в который будет сохранена фотография
    await photo_path.download(photo_file_path)

    cloth_imgs, cloth_info = get_clothes(photo_file_path)
    
    
    # Сохранение изображений в папку
    saved_image_paths = []

# Обработка каждой вещи в cloth_info
    for cloth_img, label in zip(cloth_imgs, cloth_info):

        # Сохранение изображений в папку
        folder_boxes = 'boxes'
        if not os.path.exists(folder_boxes):
            os.makedirs(folder_boxes)
        image_path = f"{folder_boxes}/{label}.jpg"
        cloth_img.save(image_path)
        # saved_image_paths.append(image_path)
        similar_images, similar_distances = find_similar_images(image_path, label, top_k=1)
        saved_image_paths.append(similar_images)

    flattened_paths = list(itertools.chain.from_iterable(saved_image_paths))
    temp_basket2 = df[df['image_url_name'].isin(flattened_paths)] #тут исправить на image_url_t_name!!!!!!

    response2 = "Рекомендации:\n\n"
    images2 = []

    for _, row in temp_basket2.iterrows():
        name = row['name']
        brand = row['brand']
        price = row['price']
        page_url = row['page_url']
        response2 += f"<a href='{page_url}'>{name}</a>\n{brand}\n{price}\n\n"
        image_url = row['image_url']
        image_data = await fetch_image(image_url)

        # Append the image bytes to the list
        images2.append(image_data)

    media_group = []
    for image_bytes in images2:
        image_file = io.BytesIO(image_bytes)
        media_group.append(types.InputMediaPhoto(image_file))
    await bot.send_media_group(message.chat.id, media=media_group)

    images2.clear()
    temp_basket2 = pd.DataFrame()

    # Reset the state to restart the process
    await state.reset_state()

    await message.reply(response2, parse_mode='HTML', disable_web_page_preview=True)
    await message.reply("Выбери как бы ты хотела подобрать образ:",
                        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                            [InlineKeyboardButton(text="По фотографии", callback_data="photo")],
                            [InlineKeyboardButton(text="По тексту", callback_data="text")],
                        ]))
    await StyleChoice.process_style.set()


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
