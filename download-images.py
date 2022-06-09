# Extracting CSV rows
import json
import os

from tqdm import tqdm
from utils import get_image_file

print('Starting CSV parse')
image2url = json.load(open('assets/image2url.json'))

def main():
    print('Starting image download')
    images_directory = os.path.join(os.path.dirname(__file__), 'assets/images')
    if not os.path.exists(images_directory):
        os.makedirs(images_directory)
    for idx, (image_name, image_url) in tqdm(enumerate(image2url.items()), desc='Downloading images...',
                                             total=len(image2url)):
        image_path = os.path.join(images_directory, f"{image_name}.png")
        image = get_image_file(image_name, image_url)
        image.save(image_path)
    print('Image download finished')

if __name__ == '__main__':
    main()
