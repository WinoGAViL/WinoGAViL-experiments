import os

TEST = 'test'
DEV = 'dev'
TRAIN = 'train'

SPLIT_PATH = os.path.join(os.path.dirname(__file__), 'assets/splits')
SWOW_DATA_PATH = os.path.join(os.path.dirname(__file__), 'assets/gvlab_swow_split.csv')
SWOW_SPLIT_PATH = os.path.join(os.path.dirname(__file__), 'assets/swow_split.json')

GAME_5_6_DATA_PATH = os.path.join(os.path.dirname(__file__), 'assets/gvlab_game_split_5_6.csv')
GAME_5_6_SPLIT_PATH = os.path.join(os.path.dirname(__file__), 'assets/game_split_5_6.json')

GAME_10_12_DATA_PATH = os.path.join(os.path.dirname(__file__), 'assets/gvlab_game_split_10_12.csv')
GAME_10_12_SPLIT_PATH = os.path.join(os.path.dirname(__file__), 'assets/game_split_10_12.json')

# IMAGES_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'assets/images')
# IMAGES_FOLDER_PATH = '/Users/yonatab/data/image_associations/gvlab-bucket'
IMAGES_FOLDER_PATH = '/data/users/yonatab/ImageAssociations/data/gvlab-bucket'
MODEL_RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'models_results')
TRAIN_RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'models_results/train')

