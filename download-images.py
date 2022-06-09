# Extracting CSV rows
import json
import os

import requests
from PIL import Image
from tqdm import tqdm

from utils import get_alternative_url

print('Starting CSV parse')
image2url = json.load(open('assets/image2url.json'))

print('Starting image download')

images_directory = os.path.join(os.path.dirname(__file__), 'assets/images')
if not os.path.exists(images_directory):
    os.makedirs(images_directory)

exceptions = []
for idx, (image_name, image_url) in tqdm(enumerate(image2url.items()), desc='Downloading images...', total=len(image2url)):
    image_path = os.path.join(images_directory, f"{image_name}.png")

    try:
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    except:
        image = Image.open(requests.get(get_alternative_url(image_name), stream=True).raw).convert("RGB")
        exceptions.append(image_name)
        print(f"Total exceptions {len(exceptions)}")
    image.save(image_path)

print('Image download finished')
