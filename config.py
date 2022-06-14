import os

TEST = 'test'
DEV = 'dev'
TRAIN = 'train'

SPLIT_PATH = os.path.join(os.path.dirname(__file__), 'assets/splits')
SWOW_DATA_PATH = os.path.join(os.path.dirname(__file__), 'assets/swow.csv')
SWOW_SPLIT_PATH = os.path.join(os.path.dirname(__file__), 'assets/cue_image_pairs_swow.json')

GAME_5_6_DATA_PATH = os.path.join(os.path.dirname(__file__), 'assets/game_5_6.csv')
GAME_5_6_SPLIT_PATH = os.path.join(os.path.dirname(__file__), 'assets/cue_image_pairs_game_5_6.json')

GAME_10_12_DATA_PATH = os.path.join(os.path.dirname(__file__), 'assets/game_10_12.csv')
GAME_10_12_SPLIT_PATH = os.path.join(os.path.dirname(__file__), 'assets/cue_image_pairs_game_10_12.json')

IMAGES_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'assets/images')
MODEL_RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'models_results')
TRAIN_RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'models_results/train')
IMAGE_CAPTIONS_PATH = os.path.join(os.path.dirname(__file__), 'assets/ofa_image_caption_predictions.csv')

columns_to_serialize = ['associations', 'distractors', 'labels', 'candidates', 'candidates_connectivity_data', 'alternative_associations_candidates']

zero_shot_results_path = os.path.join('models_results', 'zero_shot_results.csv')

