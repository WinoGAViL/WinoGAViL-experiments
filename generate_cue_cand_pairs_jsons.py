import json
import pandas as pd

from config import SWOW_SPLIT_PATH, SWOW_DATA_PATH, GAME_10_12_SPLIT_PATH, GAME_5_6_SPLIT_PATH, GAME_5_6_DATA_PATH, \
    GAME_10_12_DATA_PATH


def main():
    prepare_split(SWOW_DATA_PATH, SWOW_SPLIT_PATH)
    prepare_split(GAME_5_6_DATA_PATH, GAME_5_6_SPLIT_PATH)
    prepare_split(GAME_10_12_DATA_PATH, GAME_10_12_SPLIT_PATH)


def prepare_split(input_csv, output_json):
    ################## Loading CSV ####################
    df = pd.read_csv(input_csv)
    df['candidates'] = df['candidates'].apply(json.loads)
    df['associations'] = df['associations'].apply(json.loads)
    data = df[['ID', 'cue', 'candidates', 'associations']].to_dict('records')

    ################## Transforming CSV ######################
    cache = set()
    split_rows = []

    def get_row_id(image, cue):
        return image + '-' + cue

    def insert_row(image, cue, label, ID):
        id = get_row_id(image, cue)
        if id not in cache:
            split_rows.append({'image': image + '.jpg', 'cue': cue, 'label': label, 'ID': ID})
            cache.add(id)

    for row in data:
        cue = row['cue']
        for image in row['associations']:
            insert_row(image, cue, 1, row['ID'])
        unassociated_images = set(row['candidates']) - set(row['associations'])
        for image in unassociated_images:
            insert_row(image, cue, 0, row['ID'])
    json.dump(split_rows, open(output_json, 'w+'), indent=4)
    print(f"Wrote {len(split_rows)} (generated from {len(df)} associations) to {output_json}")


if __name__ == '__main__':
    main()
