import torch
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher import FSMContext
from aiogram.utils.exceptions import MessageNotModified
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters.state import State, StatesGroup
import aiohttp
import os
from PIL import Image
import io
from ultralytics import YOLO
import torchvision.transforms as transforms
import torchvision.models as models
import itertools

from googletrans import Translator
import emoji

import warnings
warnings.filterwarnings("ignore")

TOKEN = 'your token'
bot = Bot(token=TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot=bot, storage=storage)

# strings
GREETING = 'Привет! Я твой Personal Shopper. Давай подберем новый гардероб. Пришли мне фотографию желаемого образа или опиши его.'
LETS_FIND = 'Давай подберем новый образ. Загрузи фото или опиши желаемый образ.'
DID_NOT_FIND = 'К сожалению, я не нашел ничего подходящего :(.'
translation_table = str.maketrans({":": " ", "_": " "})

# Language state
class MyStateGroup(StatesGroup):
    lan = State()
    lets_find = State()
    did_not_find = State()

# Для текста
model_name = "cointegrated/rubert-tiny2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
df = pd.read_csv('csv/tsum_cat.csv')
categories = ['up', 'bottom', 'full', 'shoes', 'acc', 'bag']
length = 256
with open("embeddings_256.pkl", "rb") as f:
    embeddings_text = pickle.load(f)
translator = Translator(service_urls=['translate.google.com'])
# translator = Translator(service_urls=['translate.googleapis.com'])

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

def translate(text : str, lan) :
    return text if lan == 'ru' else translator.translate(text, dest=lan).text

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
        # conf = round(box.conf[0].item(), 2)

        cloth_img = image.crop(tuple(cords))
        cloth_imgs.append(cloth_img)

        cloth_info.append(label)

    return cloth_imgs, cloth_info


preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
        # transforms.RandomRotation(15),
        # transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
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

async def init_lan(state : FSMContext) :
    await state.update_data(lan='ru')
    await state.update_data(lets_find = LETS_FIND)
    await state.update_data(did_not_find = DID_NOT_FIND)

@dp.message_handler(commands=['start'])
async def start(message: types.Message, state: FSMContext):
    init_lan(state)
    await message.reply(GREETING)

async def process_basket(message : types.Message, 
                         state : FSMContext,
                         basket_df : pd.DataFrame) :
    response = ""
    images = []

    data = await state.get_data()
    if len(data) == 0 : 
        await init_lan(state)
        data = await state.get_data()

    for _, row in basket_df.iterrows():
        name = translate(row['name'], data['lan'])
        brand = row['brand']
        page_url = row['page_url']
        response += f"<a href='{page_url}'>{name}</a>\n{brand}\n\n"
        image_url = row['image_url']
        image_data = await fetch_image(image_url)

        # Append the image bytes to the list
        images.append(image_data)

    if len(images) > 0 :
        media_group = []
        for image_bytes in images:
            image_file = io.BytesIO(image_bytes)
            media_group.append(types.InputMediaPhoto(image_file))
        await bot.send_media_group(message.chat.id, media=media_group)

        images.clear()

        # Reply with a new message
        await message.reply(response, 
                            parse_mode='HTML', 
                            disable_web_page_preview=True)
    
    else :
        await message.reply(data['did_not_find'])

@dp.message_handler(content_types=types.ContentType.TEXT)
async def process_text(message: types.Message, state: FSMContext):

    text = emoji.demojize(message.text.strip()).translate(translation_table)
    query = translator.translate(text, dest='ru')

    # Сохраняем язык в переменной lan (если он изменился)
    data = await state.get_data()
    if len(data) == 0 : 
        await init_lan(state)
        data = await state.get_data()
        

    lets_find = data['lets_find']

    if data['lan'] != query.src :
        await state.update_data(lan = query.src)
        did_not_find = translate(DID_NOT_FIND, query.src)
        lets_find = translate(LETS_FIND, query.src)
        await state.update_data(did_not_find = did_not_find)
        await state.update_data(lets_find = lets_find) 

    basket_df = generate_recommendations(query.text, embeddings_text)
    await process_basket(message, state, basket_df)
    await message.bot.send_message(message.from_user.id, lets_find)


@dp.message_handler(content_types=types.ContentType.PHOTO)
async def process_photo(message: types.Message, state: FSMContext):
    photo = message.photo[-1]
    photo_id = photo.file_id
    photo_path = await bot.get_file(photo_id)
    chat_id = message.chat.id
    
    # Загрузка фотографии и сохранение ее на диске
    folder_temporary = 'temporary_request'
    if not os.path.exists(folder_temporary):
        os.makedirs(folder_temporary)
    photo_file_path = f"{folder_temporary}/{chat_id}_photo.jpg"  # Путь к файлу, в который будет сохранена фотография
    await photo_path.download(photo_file_path)

    data = await state.get_data() 
    if len(data) == 0 : 
        await init_lan(state)
        data = await state.get_data()

    cloth_imgs, cloth_info = get_clothes(photo_file_path)

    if len(cloth_imgs) > 0 :  
  
        # Сохранение изображений в папку
        saved_image_paths = []

        # Обработка каждой вещи в cloth_info
        for cloth_img, label in zip(cloth_imgs, cloth_info):

            # Сохранение изображений в папку
            folder_boxes = 'boxes'
            if not os.path.exists(folder_boxes):
                os.makedirs(folder_boxes)
            image_path = f"{folder_boxes}/{chat_id}_{label}.jpg"
            cloth_img.save(image_path)
            # saved_image_paths.append(image_path)
            similar_images, similar_distances = find_similar_images(image_path, label, top_k=1)
            saved_image_paths.append(similar_images)

        flattened_paths = list(itertools.chain.from_iterable(saved_image_paths))
        basket_df = df[df['image_url_name'].isin(flattened_paths)] #тут исправить на image_url_t_name!!!!!!
        await process_basket(message, state, basket_df)
    else :
        await message.reply(data['did_not_find'])
    
    await message.bot.send_message(message.from_user.id, data['lets_find'])


# handle the cases when this exception raises 
@dp.errors_handler(exception=MessageNotModified)  
async def message_not_modified_handler(update, error): 
    return True # errors_handler must return True if error was handled correctly 

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
