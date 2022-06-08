import os
from collections import defaultdict

import clip
import pandas as pd
import spacy
import timm
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from timm.data import resolve_data_config, create_transform
from transformers import ViltProcessor, ViltForImageAndTextRetrieval, RobertaTokenizer, AutoFeatureExtractor
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    ViTFeatureExtractor,
    BertTokenizer,
)

from dataset.config import images_path, image_captions_path
from models_config import MODELS_MAP, VISION_LANGUAGE_MODELS, VISION_MODELS, TEXT_TRANSFORMERS_MODELS

nlp = spacy.load('en_core_web_md')
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

if os.path.exists(image_captions_path):
    image_captions = pd.read_csv(image_captions_path)

all_images_paths = []
missing_images = []

class WinoGAViLZeroShotModel:
    def __init__(self, model_description, image2text):
        self.image2text = image2text
        self.model_description = model_description
        self.model_factory(model_description)

    def model_factory(self, model_description):
        requires_freeze = False
        if 'clip' in model_description.lower():
            self.model, self.preprocess_func = clip.load(MODELS_MAP[model_description], device=device)
            requires_freeze = True
        elif model_description == 'ViLT':
            requires_freeze = True
            self.model = ViltForImageAndTextRetrieval.from_pretrained(MODELS_MAP[model_description])
            self.model.to(device)
            self.preprocess_func = ViltProcessor.from_pretrained(MODELS_MAP[model_description])
        elif 'LiT' in model_description:
            requires_freeze = True
            vision_backend, text_backend = MODELS_MAP[model_description]['vision'], MODELS_MAP[model_description][
                'text']
            feature_extractor, tokenizer = self.get_lit_tokenizers(text_backend, vision_backend)
            self.preprocess_func = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)
            self.model = VisionTextDualEncoderModel.from_vision_text_pretrained(
                vision_backend, text_backend
            )
            self.model.to(device)
        elif model_description in VISION_MODELS.keys():
            requires_freeze = True
            self.model, self.preprocess_func = self.create_timm_model(MODELS_MAP[model_description])
        elif model_description.lower() == 'word2vec':
            pass
        elif model_description in TEXT_TRANSFORMERS_MODELS.keys():
            self.model = SentenceTransformer(MODELS_MAP[model_description])
        else:
            raise Exception(f'Unexpected model: {model_description}')

        if requires_freeze:
            total_parameters = sum(p.numel() for p in self.model.parameters())
            print(f"model: {self.model_description}, # parameters: {total_parameters}, freezing...")
            for param in self.model.parameters():
                param.requires_grad = False

    def get_lit_tokenizers(self, text_backend, vision_backend):
        if 'roberta' in text_backend:
            tokenizer = RobertaTokenizer.from_pretrained(text_backend)
        else:
            tokenizer = BertTokenizer.from_pretrained(text_backend)
        if 'vit' in vision_backend:
            feature_extractor = ViTFeatureExtractor.from_pretrained(vision_backend)
        else:
            feature_extractor = AutoFeatureExtractor.from_pretrained(vision_backend)
        return feature_extractor, tokenizer

    def create_timm_model(self, backend_version):
        model = timm.create_model(backend_version, pretrained=True).to(device)
        model.eval()
        vit_config = resolve_data_config({}, model=model)
        transform = create_transform(**vit_config)
        return model, transform

    def get_predictions(self, candidates_data, cue_img, r):
        if 'clip' in self.model_description.lower():
            clip_cue_img, clip_cue_txt = self.get_clip_txt_and_img(r['cue'], cue_img)
            model_predictions = self.calculate_clip_scores(candidates_data, clip_cue_img, clip_cue_txt)
        elif self.model_description in VISION_LANGUAGE_MODELS.keys():
            model_predictions = self.get_vl_transformers_predictions(r, candidates_data, cue_img)
        elif self.model_description in VISION_MODELS.keys():
            model_predictions = self.get_vision_predictions(candidates_data, cue_img)
        elif self.model_description.lower() == 'word2vec':
            model_predictions = self.get_word2vec_predictions(r)
        elif self.model_description in TEXT_TRANSFORMERS_MODELS.keys():
            cue_candidates_combinations, candidates_words = self.get_cue_candidates_combinations(candidates_data, cue_img, r)
            model_predictions = {}
            for combination, cue_cand_data in cue_candidates_combinations.items():
                cue, candidates = cue_cand_data[0], cue_cand_data[1]
                combination_model_predictions = self.get_sentence_similarity_predictions(combination, cue, candidates, candidates_words)
                model_predictions = {**model_predictions, **combination_model_predictions}
        else:
            raise Exception(f"Unexpected model: {self.model_description}")
        return model_predictions

    def get_cue_candidates_combinations(self, candidates_data, cue_img, r):
        cue_word = r['cue']
        candidates_words = [x['distractor'] if type(x) == dict else x for x in r['candidates']]
        cue_candidates_combinations = {'cue_cand_words': (cue_word, candidates_words)}
        if self.image2text:
            candidates_image_captions = [x['cand_img'] for x in candidates_data]
            image_captions_combinations = {'cue_word_cand_img_caption': (cue_word, candidates_image_captions),
                                           # 'cue_img_caption_cand_word': (cue_img, candidates_words),
                                           # 'cue_cand_img_captions': (cue_img, candidates_image_captions)
                                           }
            cue_candidates_combinations = {**cue_candidates_combinations, **image_captions_combinations}
        return cue_candidates_combinations, candidates_words

    def get_vision_predictions(self, candidates_data, cue_img):
        model_predictions = defaultdict(dict)
        cue_img_features = self.get_timm_img_feats(cue_img)
        for cand_data in candidates_data:
            cand_img_features = self.get_timm_img_feats(cand_data['cand_img'])
            cue_cand_img_sim = self.get_vectors_similarity(cue_img_features, cand_img_features)
            model_predictions[f'timm_{self.model_description}_cue_cand_img'][cand_data['txt']] = cue_cand_img_sim.item()
        return model_predictions

    def get_timm_img_feats(self, img):
        timm_img = self.preprocess_timm_img(img)
        timm_img_features = self.forward_timm_model(timm_img)
        timm_img_features /= timm_img_features.norm(dim=-1, keepdim=True)
        return timm_img_features

    def get_word2vec_predictions(self, r):
        model_predictions = defaultdict(dict)
        candidates = [x['distractor'] if type(x) == dict else x for x in r['candidates']]
        for cand in candidates:
            sim_cue_cand_spacy = nlp(r['cue']).similarity(nlp(cand))
            model_predictions['word2vec_cue_cand'][cand] = sim_cue_cand_spacy
        return dict(model_predictions)

    def get_sentence_similarity_predictions(self, combination, cue, candidates, candidates_words):
        model_predictions = defaultdict(dict)
        encoded_cue = self.preprocess_sentence_txt(cue)
        for cand, cand_word in zip(candidates, candidates_words):
            encoded_cand = self.preprocess_sentence_txt(cand)
            sim_cue_cand_bert = cosine_similarity([encoded_cue], [encoded_cand])[0][0]
            model_predictions[f'{combination}_{self.model_description}'][cand_word] = sim_cue_cand_bert
        return dict(model_predictions)

    def calculate_clip_scores(self, candidates_data, cue_clip_img_encoded, cue_clip_txt_encoded):
        all_cue_txt_cand_sim = defaultdict(dict)
        for cand_data in candidates_data:
            cand_clip_img_encoded, cand_clip_txt_encoded = self.get_clip_txt_and_img(cand_data['cand_txt'], cand_data['cand_img'])

            cue_txt_cand_txt_sim = self.get_vectors_similarity(cue_clip_txt_encoded, cand_clip_txt_encoded)
            cue_txt_cand_img_sim = self.get_vectors_similarity(cue_clip_txt_encoded, cand_clip_img_encoded)
            all_cue_txt_cand_sim[f'{self.model_description}_cue_txt_cand_txt'][cand_data['txt']] = cue_txt_cand_txt_sim.item()
            all_cue_txt_cand_sim[f'{self.model_description}_cue_txt_cand_img'][cand_data['txt']] = cue_txt_cand_img_sim.item()

            if type(cue_clip_img_encoded) != type(None):
                cue_img_cand_txt_sim = self.get_vectors_similarity(cue_clip_img_encoded, cand_clip_txt_encoded)
                cue_img_cand_img_sim = self.get_vectors_similarity(cue_clip_img_encoded, cand_clip_img_encoded)
                all_cue_txt_cand_sim[f'{self.model_description}_cue_img_cand_txt'][cand_data['txt']] = cue_img_cand_txt_sim.item()
                all_cue_txt_cand_sim[f'{self.model_description}_cue_img_cand_img'][cand_data['txt']] = cue_img_cand_img_sim.item()
        return all_cue_txt_cand_sim

    def get_vl_transformers_predictions(self, r, candidates_data, cue_img):
        all_cue_txt_cand_sim = defaultdict(dict)
        for cand_data in candidates_data:
            cue_txt_cand_img_sim = self.single_vl_transformer_score(r['cue'], cand_data['cand_img'])
            all_cue_txt_cand_sim[f'{self.model_description}_cue_txt_cand_img'][cand_data['txt']] = cue_txt_cand_img_sim
            if type(cue_img) != type(None):
                cue_img_cand_txt_sim = self.single_vl_transformer_score(cand_data['cand_txt'], cue_img)
                all_cue_txt_cand_sim[f'{self.model_description}_cue_img_cand_txt'][cand_data['txt']] = cue_img_cand_txt_sim
        return all_cue_txt_cand_sim

    def single_vl_transformer_score(self, text, image):
        if 'ViLT' in self.model_description:
            encoding = self.preprocess_func(image, text, return_tensors="pt").to(device)
            outputs = self.model(**encoding)
            score = outputs.logits[:,0].item()
        elif 'LiT' in self.model_description:
            encoding = self.preprocess_func(images=image, text=[text], return_tensors="pt", padding=True).to(device)
            outputs = self.model(input_ids=encoding.input_ids,
                                     attention_mask=encoding.attention_mask,
                                     pixel_values=encoding.pixel_values,
                                     return_loss=True,
                                     )
            loss, logits_per_image = outputs.loss, outputs.logits_per_image  # this is the image-text similarity score
            score = logits_per_image.item()
        else:
            raise Exception(f"Unexpected Model: {self.model_description}")
        return score

    def get_clip_txt_and_img(self, txt, img):
        if self.image2text:
            cue_clip_img_caption = self.get_clip_txt(img)
            cue_clip_img_encoded = self.model.encode_text(cue_clip_img_caption)
            cue_clip_img_encoded /= cue_clip_img_encoded.norm(dim=-1, keepdim=True)
        else:
            if img:
                cue_clip_img = self.preprocess_clip_img(img)
                cue_clip_img_encoded = self.model.encode_image(cue_clip_img)
                cue_clip_img_encoded /= cue_clip_img_encoded.norm(dim=-1, keepdim=True)
            else:
                cue_clip_img, cue_clip_img_encoded = None, None
        cue_clip_txt = self.get_clip_txt(txt)
        cue_clip_txt_encoded = self.model.encode_text(cue_clip_txt)
        cue_clip_txt_encoded /= cue_clip_txt_encoded.norm(dim=-1, keepdim=True)
        return cue_clip_img_encoded, cue_clip_txt_encoded


    def get_vectors_similarity(self, v1, v2):
        similarity = v1.detach().cpu().numpy() @ v2.detach().cpu().numpy().T
        return similarity


    def preprocess_clip_img(self, img):
        if self.image2text:
            return img
        clip_img = self.preprocess_func(img).unsqueeze(0).to(device)
        return clip_img

    def get_clip_txt(self, item):
        item = item.lower()
        vowels = ['a', 'e', 'i', 'o', 'u']
        if any(item.startswith(x) for x in vowels):
            clip_txt = f"An {item}"
        else:
            clip_txt = f"A {item}"
        clip_txt_tokenized = clip.tokenize([clip_txt]).to(device)
        return clip_txt_tokenized

    def forward_timm_model(self, img):
        x = self.model.forward_features(img)
        if self.model_description.lower() == 'convnext':
            x = self.model.head(x)
        return x

    def preprocess_timm_img(self, img):
        img_preprocessed = self.preprocess_func(img).unsqueeze(0)
        img_preprocessed = img_preprocessed.to(device)
        return img_preprocessed

    def preprocess_sentence_txt(self, txt):
        v = self.model.encode(txt)
        return v

    @staticmethod
    def get_img(cand, image2text=False):
        cand_path = os.path.join(images_path, f"{cand}.jpg")
        if os.path.exists(cand_path):
            global all_images_paths
            all_images_paths.append(cand_path)
            if image2text:
                # relevant_caption_rows = image_captions[image_captions['img_path'] == cand_path]['caption']
                relevant_caption_rows = image_captions[image_captions['img_path'].apply(lambda x: x.split("/")[-1]) == cand_path.split("/")[-1]]['caption']
                try:
                    assert len(relevant_caption_rows) == 1
                except:
                    global missing_images
                    missing_images.append(cand_path.split("/")[-1])
                    return None
                image_caption = relevant_caption_rows.iloc[0]
                return image_caption
            img = Image.open(cand_path).convert("RGB")
            return img
        return None