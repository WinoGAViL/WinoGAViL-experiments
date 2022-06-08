# ------------------------------Imports--------------------------------
import json

import torch
import numpy as np
import os
import pandas as pd
import pickle

# ------------------------------Code--------------------------------
from config import TEST, DEV, TRAIN, SWOW_SPLIT_PATH, MODEL_RESULTS_PATH, TRAIN_RESULTS_PATH



def calculate_accuracy(out_prob, y):
    prob = torch.softmax(out_prob, dim=1)
    out_np = prob.detach().cpu().numpy()
    labels_np = y.detach().cpu().numpy()
    accuracy = (np.argmax(out_np, 1) == labels_np).mean()
    predictions = [float(x) for x in np.argmax(out_np, 1)]
    labels = [float(x) for x in labels_np]
    return accuracy, predictions, labels


def save_model(model_dir_path, epoch, model):
    out_p = os.path.join(model_dir_path, f"epoch_{epoch}.pth")
    print(f"Saving model path to... {out_p}")
    torch.save(model.state_dict(), out_p)
    # if dev_accuracy_list[-1] == max(dev_accuracy_list):
    #     out_p = os.path.join(model_dir_path, f"epoch_BEST.pth")
    #     print(f"Saving BEST model path to... {out_p}")
    #     print(dev_accuracy_list)
    #     torch.save(model.state_dict(), out_p)


# def dump_test_info(args, model_dir_path, all_losses, all_test_accuracy, test_df, epoch):
#     test_losses_mean = {i: np.mean(v) for i, v in enumerate(all_losses['test'])}
#     test_accuracy_mean = {i: np.mean(v) for i, v in enumerate(all_test_accuracy)}
#     test_info = pd.concat(
#         [pd.Series(test_losses_mean, name='test loss'), pd.Series(test_accuracy_mean, name='test accuracy')], axis=1)
#     out_p = os.path.join(model_dir_path, f'epoch_{epoch}_test')
#     if args.result_suffix != '':
#         out_p += "_" + args.result_suffix
#     all_losses_out_p = out_p + '_all_losses_test.pickle'
#     out_p_test_df = out_p + "_test_df.csv"
#     out_p += ".csv"
#     test_info.to_csv(out_p)
#     test_df.to_csv(out_p_test_df)
#     all_losses_and_acc_d = {'all_losses': all_losses, 'all_test_accuracy': all_test_accuracy}
#     with open(all_losses_out_p, 'wb') as f:
#         pickle.dump(all_losses_and_acc_d, f)
#     print(f'Dumping losses {len(test_info)} to {all_losses_out_p}')
#     print(test_info)
#     print(f'Dumping df {len(test_info)} to {out_p}, and {len(test_df)} to {out_p_test_df}')


def dump_train_info(args, model_dir_path, all_losses, all_dev_accuracy, epoch):
    train_losses_mean = {i: np.mean(v) for i, v in enumerate(all_losses['train'])}
    dev_losses_mean = {i: np.mean(v) for i, v in enumerate(all_losses['dev'])}
    dev_accuracy_mean = {i: np.mean(v) for i, v in enumerate(all_dev_accuracy)}
    train_info = pd.concat(
        [pd.Series(train_losses_mean, name='train loss'), pd.Series(dev_losses_mean, name='dev loss'),
         pd.Series(dev_accuracy_mean, name='dev accuracy')], axis=1)
    out_p = os.path.join(model_dir_path, f'epoch_{epoch}')
    if args.result_suffix != '':
        out_p += "_" + args.result_suffix
    all_losses_out_p = out_p + '_all_losses.pickle'
    out_p += ".csv"
    train_info.to_csv(out_p)
    dev_loss_list = list(train_info['dev loss'].values)
    dev_accuracy_list = list(train_info['dev accuracy'].values)
    print(f"*** dev loss ***")
    print(dev_loss_list)
    print(f"*** dev accuracy ***")
    print(dev_accuracy_list)
    all_losses_and_acc_d = {'all_losses': all_losses, 'all_dev_accuracy': all_dev_accuracy}
    with open(all_losses_out_p, 'wb') as f:
        pickle.dump(all_losses_and_acc_d, f)
    print(f'Dumping losses {len(train_info)} to {all_losses_out_p}')
    print(train_info)
    print(f'Dumping df {len(train_info)} to {out_p}')
    return dev_accuracy_list



def get_gvlab_data(args):
    if args.test_only:
        test = get_relevant_test(args)
        # test = pd.read_csv(f'assets/gvlab_{args.split}.csv')
        # test = pd.read_csv(f'assets/gvlab_game_split_5_6.csv')
        # print("gvlab_game_split_5_6")
        # test = pd.read_csv(f'assets/gvlab_game_split_10_12.csv')
        # print("gvlab_game_split_10_12")
        # test = pd.read_csv(f'assets/gvlab_swow_split.csv')
        # print("swow_split")
        # test['candidates'] = test['candidates'].apply(json.loads)
        # test['associations'] = test['associations'].apply(json.loads)
        print(f"Got test, size {len(test)}")
        splits = {'test': test}
    else:
        if args.split == 'swow_split':
            print(f"Training on SWOW SPLIT")
            f = open(f"assets/swow_split.json")
            train = json.load(f)
        else:
            print(f"Training on Game SPLIT (5,6,10,12)")
            train_5_6 = json.load(open(f"assets/game_split_5_6.json"))
            train_10_12 = json.load(open(f"assets/game_split_10_12.json"))
            train = train_5_6 + train_10_12
            print(f"Total train size is {len(train)}")

        print(f"Reading test from {args.split}")
        # df = pd.read_csv(f'assets/gvlab_{args.split}.csv')
        df = pd.read_csv(f'assets/test_sets_with_zero_shot_predictions/gvlab_{args.split}_with_predictions.csv')
        print(f"Split: {args.split}, read data with predictions, mean jaccard: {df['clip_vit_32_jaccard'].mean()}")
        df['candidates'] = df['candidates'].apply(json.loads)
        df['associations'] = df['associations'].apply(json.loads)

        items_in_test_dev = int(len(df) * args.dev_test_sample)
        test = df.sample(items_in_test_dev)

        all_test_candidates = get_image_candidates_set(test)
        all_test_dev_candidates, dev = get_dev_without_test_images(all_test_candidates, df, items_in_test_dev, test)

        dev_unique_ids, test_unique_ids, train, train_unique_ids = get_train_without_testdev_images(
            all_test_dev_candidates, dev, test, train)
        print(f"train: {len(train)}, # {len(train_unique_ids)} unique IDs")
        # print(f"dev: {len(dev)}, # {len(dev_unique_ids)} unique IDs")
        # print(f"test: {len(test)}, # {len(test_unique_ids)} unique IDs")
        print(f"dev: {len(dev)}, # {len(dev_unique_ids)} unique IDs, Jaccard: {round(dev['clip_vit_32_jaccard'].mean() * 100 , 1)}")
        print(f"test: {len(test)}, # {len(test_unique_ids)} unique IDs, Jaccard: {round(test['clip_vit_32_jaccard'].mean() * 100 , 1)}")

        splits = {'train': train, 'dev': dev, 'test': test}
    return splits


def get_train_without_testdev_images(all_test_dev_candidates, dev, test, train):
    train_not_intersected_with_test_candidates = []
    for x in train:
        if x['image'].split(".")[0] not in all_test_dev_candidates:
            train_not_intersected_with_test_candidates.append(x)
    print(f"Started with {len(train)} in train, not intersected are {len(train_not_intersected_with_test_candidates)} ")
    train = train_not_intersected_with_test_candidates
    all_train_candidates = set([x['image'] for x in train])
    assert len(all_train_candidates.intersection(all_test_dev_candidates)) == 0
    excluded_ids = set(test['ID'].values).union(set(dev['ID'].values))
    train = [x for x in train if x['ID'] not in excluded_ids]
    test_unique_ids = set(test['ID'])
    dev_unique_ids = set(dev['ID'])
    train_unique_ids = set([x['ID'] for x in train])
    assert len(test_unique_ids & dev_unique_ids & train_unique_ids) == 0
    return dev_unique_ids, test_unique_ids, train, train_unique_ids


def get_dev_without_test_images(all_test_candidates, df, items_in_test_dev, test):
    df = df[~df['ID'].isin(test['ID'])]
    dev_items = []
    for r_idx, r in df.iterrows():
        row_not_in_test_images = all(cand not in all_test_candidates for cand in r['candidates'])
        if row_not_in_test_images:
            dev_items.append(r)
        if len(dev_items) >= items_in_test_dev:
            break
    dev = pd.DataFrame(dev_items)
    all_dev_candidates = get_image_candidates_set(dev)
    assert len(all_dev_candidates.intersection(all_test_candidates)) == 0
    all_test_dev_candidates = all_dev_candidates.union(all_test_candidates)
    return all_test_dev_candidates, dev


def get_image_candidates_set(test_dev_data):
    all_test_dev_candidates = []
    for cand in test_dev_data['candidates']:
        all_test_dev_candidates += cand
    all_test_dev_candidates = set(all_test_dev_candidates)
    return all_test_dev_candidates


def get_relevant_test(args):
    model_experiment_dir = get_experiment_dir(args)
    test_path = os.path.join(model_experiment_dir, 'splits', 'test.csv')
    test = pd.read_csv(test_path)
    # test['candidates'] = test['candidates'].apply(json.loads)
    # test['associations'] = test['associations'].apply(json.loads)
    test['candidates'] = test['candidates'].apply(lambda x: json.loads(x.replace("'", '"')))
    test['associations'] = test['associations'].apply(lambda x: json.loads(x.replace("'", '"')))
    return test


def get_experiment_dir(args):
    if not os.path.exists(MODEL_RESULTS_PATH):
        os.makedirs(MODEL_RESULTS_PATH)
    if not os.path.exists(TRAIN_RESULTS_PATH):
        os.makedirs(TRAIN_RESULTS_PATH)

    model_dir_path = os.path.join(TRAIN_RESULTS_PATH, f"model_backend_{args.model_backend_type.replace('/','-')}_{args.model_version.replace('/', '-')}_{args.split}_{args.experiment_idx}")
    # model_dir_path = os.path.join(TRAIN_RESULTS_PATH, f"model_backend_{args.model_backend_type.replace('/','-')}_{args.model_version.replace('/', '-')}_{args.split}")

    if args.debug:
        model_dir_path += "_DEBUG"
    if not os.path.exists(model_dir_path):
        os.mkdir(model_dir_path)
    json.dump(args.__dict__, open(os.path.join(model_dir_path, 'args.json'), 'w'))
    return model_dir_path
