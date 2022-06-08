import argparse
import os
import json
from collections import Counter

import pandas as pd
from tqdm import tqdm
from models_config import *

from AssociationModel import AssociationsModel
from dataset.config import columns_to_serialize, \
    concreteness_path, \
    concreteness_threshold, baseline_all_models_path, all_loaded_images_path

all_images_paths = []

def main(args):
    df = pd.read_csv(f'assets/gvlab_{args.split}.csv')
    print(f"SPLIT: {args.split}, Read dataset at length: {len(df)}, from: {dataset_path}")

    for c in columns_to_serialize:
        if c in df:
            df[c] = df[c].apply(json.loads)

    missing_images_indices = []
    items_with_predictions = []
    missing_candidates_images = []
    for idx, (r_idx, r) in enumerate(tqdm(df.iterrows(), desc=f'Solving Dataset ({args.split})', total=len(df))):
        print_stats(idx, items_with_predictions, missing_images_indices)

        candidates, candidates_data, missing_cand_image = get_candidates_data(r, image2text=args.image2text)
        if missing_cand_image:
            print(f"missing_candidates_images: {len(missing_candidates_images)} += 1")
            missing_candidates_images.append(idx)


        if args.multimodal:
            cue_img = None
            if missing_cand_image or type(cue_img) == type(None):
                missing_images_indices.append(idx)
                continue
        else:
            cue_img = AssociationsModel.get_img(r['cue'], image2text=args.image2text)

        row_predictions = {}
        for model_name, model in association_models.items():
            model_predictions = model.get_predictions(candidates_data, cue_img, r)
            row_predictions = {**row_predictions, **model_predictions}

        choose_best_candidates(row_predictions, candidates, items_with_predictions, r)

    mean_scores, scores_df = get_scores(items_with_predictions)
    print(f"Jaccard Results")
    print(mean_scores)

    if args.multimodal:
        out_p = baseline_all_models_path.replace(".csv",f"_{args.split}_multimodal.csv")
    else:
        out_p = baseline_all_models_path.replace(".csv",f"_{args.split}.csv")


    print(f"Writing predictions: {len(scores_df)} to {out_p}")
    scores_df.to_csv(out_p, index=False)

    print(f"missing_images: {len(missing_images_indices)}")

    print(f"Writing all_images_paths ({len(all_images_paths)}) to: {all_loaded_images_path}")
    all_loaded_images_path_out_p = all_loaded_images_path.replace(".csv",f"_{args.split}.csv")
    pd.DataFrame(list(set(all_images_paths))).to_csv(all_loaded_images_path_out_p)

    print("Done")


def choose_best_candidates(all_cue_txt_cand_sim, candidates, items_with_predictions, r):
    predictions_for_modality = {}
    for prediction_type, items in all_cue_txt_cand_sim.items():
        most_similar_items = Counter(items).most_common()[:r['num_associations']]
        most_similar_items_str = [x[0] for x in most_similar_items]
        predictions_for_modality[prediction_type] = most_similar_items_str
        correct_intersections = set(r['associations']).intersection(most_similar_items_str)
        assert len(most_similar_items_str) == len(r['associations'])
        pred_and_label_union = set(most_similar_items_str).union(set(r['associations']))
        jaccard = len(correct_intersections) / len(pred_and_label_union)
        r[f'predictions_{prediction_type}'] = most_similar_items_str
        r[f'correct_intersections_{prediction_type}'] = correct_intersections
        r[f'pred_and_label_union_{prediction_type}'] = pred_and_label_union
        r[f'jaccard_score_{prediction_type}'] = jaccard
        items_with_predictions.append(r)


def get_candidates_data(r, image2text):
    candidates = r['candidates']
    candidates_data = []
    missing_image = False
    for cand in candidates:
        cand_img = AssociationsModel.get_img(cand, image2text)
        if type(cand_img) == type(None):
            missing_image = True
        candidates_data.append({'txt': cand, 'cand_img': cand_img, 'cand_txt': cand})
    return candidates, candidates_data, missing_image


def print_stats(idx, items_with_predictions, missing_images_indices):
    if idx in [10, 25, 50] or (idx > 0 and idx % 100 == 0):
        mean_scores, scores_df = get_scores(items_with_predictions)
        print(f"idx: {idx}, missing_images_indices: {len(missing_images_indices)}, mean_scores:")
        print(mean_scores)


def get_scores(items_with_predictions):
    scores_df = pd.DataFrame(items_with_predictions)
    score_columns = [c for c in scores_df.columns if 'jaccard' in c]
    mean_scores = scores_df[score_columns].mean().apply(lambda x: int(x * 100))
    return mean_scores, scores_df


def initialize_models(args):
    global association_models
    association_models = {}
    for model in args.models_to_run:
        associations_model = AssociationsModel(model, image2text=args.image2text)
        association_models[model] = associations_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_to_run', default=list(GENERAL_MODELS.keys()) + list(VISION_LANGUAGE_MODELS.keys()))
    parser.add_argument('--models_to_run', default=MODELS_MAP.keys())
    # parser.add_argument('--models_to_run', default=['CLIP-RN50'])
    # parser.add_argument('--models_to_run', default=['CLIP-ViT-L/14'])
    # parser.add_argument('--split', default='swow')
    parser.add_argument('--split', default='game')
    parser.add_argument("--multimodal", action='store_const', default=False, const=True)
    parser.add_argument("--image2text", action='store_const', default=False, const=True)
    args = parser.parse_args()
    if args.image2text:
        args.models_to_run = TEXT_TRANSFORMERS_MODELS.keys()
        # args.models_to_run = ['MPNet']

    print(args)

    initialize_models(args)

    concreteness_df = pd.read_excel(concreteness_path)
    concreteness_for_word = dict(concreteness_df[['Word', 'Conc.M']].values)
    main(args)
    print(f'missing_images: {missing_images}')