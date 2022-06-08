import argparse
import json
import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch import nn
torch.autograd.set_detect_anomaly(True)

from config import TRAIN, TRAIN_RESULTS_PATH, MODEL_RESULTS_PATH, DEV, TEST
from models.gvlab_backend import BackendModel
from models.gvlab_trainable import BaselineModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import save_model, dump_train_info, get_gvlab_data, get_experiment_dir

device_ids = [0, 1, 2, 3]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--lr', help='learning rate', default=0.001, type=float)
    parser.add_argument('-bz', '--batch_size', default=128, type=int)
    # parser.add_argument('-bz', '--batch_size', default=32, type=int)
    # parser.add_argument('-bz', '--batch_size', default=4, type=int)
    parser.add_argument('-ne', '--n_epochs', default=7, type=int)
    parser.add_argument('--dev_test_sample', default=0.1, type=int)
    parser.add_argument('-s', '--split', default='gvlab_swow_split')  # gvlab_swow_split, gvlab_game_split_5_6, gvlab_game_split_10_12
    parser.add_argument('-rs', '--result_suffix', default="", required=False, help='suffix to add to results name')
    parser.add_argument("--debug", action='store_const', default=False, const=True)
    parser.add_argument("--test_model", action='store_const', default=True, const=True)
    parser.add_argument("--test_only", action='store_const', default=False, const=True)
    parser.add_argument('--load_epoch', default=0)
    parser.add_argument('--num_experiments', default=5)
    parser.add_argument('--model_backend_type', default='ViT-B/32', help="CLIP backend type", required=False)
    parser.add_argument('--model_version', default='1.0.0', help="version", required=False)
    args = parser.parse_args()
    print(args)
    if args.debug:
        print(f"*** DEBUG ***")
    return args


class Loader(Dataset):
    def __init__(self, data, backend_model, is_train=True):
        self.data = data
        self.backend_model = backend_model
        self.is_train = is_train

    def __getitem__(self, index):
        if self.is_train:
            row = self.data[index]
            text_vector = self.backend_model.encode_text(row['cue'])
            input_image_vector = self.backend_model.load_and_encode_img(row['image'])
            return input_image_vector, text_vector, row['label']
        else:
            row = self.data.iloc[index]
            text_vector = self.backend_model.encode_text(row['cue'])
            all_labels = []
            options_candidates = []
            for cand in row['candidates']:
                if cand in row['associations']:
                    label = 1
                else:
                    label = 0
                input_image_vector = self.backend_model.load_and_encode_img(cand + ".jpg")
                options_candidates.append(input_image_vector)
                all_labels.append(label)
            for i in range(12 - len(options_candidates)):
                options_candidates.append(torch.zeros(1, 512).to(device))
                all_labels.append(-1)
            return text_vector, torch.cat(options_candidates), np.array(all_labels), row['num_associations']

    def __len__(self):
        return len(self.data)


def test(backend_model, baseline_model, data):
    """
    Defines the parameters for the test loop and runs it

    Parameters
    ----------
    backend_model : BackendModel is the feature extraction model (e.g., VIT)
    baseline_model :(nn.Module) The baseline model
    data :(dict) contains the train, dev and test data
    loss_fn : Loss function

    """
    print('*** testing ***')
    test_dataset = Loader(data[TEST], backend_model, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    model_dir_path = get_experiment_dir(args)
    model_path = os.path.join(model_dir_path, f'epoch_{args.load_epoch}.pth')
    print(f"Loading model (epoch_{args.load_epoch}) from {model_path}")
    assert os.path.exists(model_path)
    baseline_model.load_state_dict(torch.load(model_path))
    baseline_model = baseline_model.eval()
    results_zeroshot_vs_trainable = test_loop(args=args, model=baseline_model, test_loader=test_loader, test_df=data[TEST])
    return results_zeroshot_vs_trainable

def test_loop(args, model, test_loader, test_df):
    """
    Runs the test loop on the given test set
    Parameters
    ----------
    args :  (argparse.Namespace) arguments
    model :(nn.Module) The baseline model
    test_loader :(DataLoader)
    loss_fn : Loss function
    test_df : (DataFrame) contain information the test set

    """
    all_losses = {TEST: []}
    all_test_accuracy = []
    model_dir_path = get_experiment_dir(args)

    epoch_test_losses, epoch_test_accuracy, predictions, labels = test_epoch(model, test_loader, args.load_epoch)
    all_losses[TEST].append(epoch_test_losses)
    all_test_accuracy.append(epoch_test_accuracy)

    test_df['predictions'] = predictions
    test_df['labels'] = labels

    results_zeroshot_vs_trainable = dump_test_info(args, model_dir_path, all_losses, all_test_accuracy, test_df, epoch=args.load_epoch)
    return results_zeroshot_vs_trainable


def test_epoch(model, dev_loader, epoch):
    """
    Tests the model on a single epoch using the given dev_loader

    Parameters
    ----------
    loss_fn : Loss function
    model : (nn.Module) The baseline model
    dev_loader : (DataLoader)
    epoch : (int) epoch number

    Returns
    -------
    The epoch: losses,accuracy,model's prediction, labels

    """

    model.eval()
    epoch_dev_losses = []
    epoch_dev_accuracy = []
    all_predictions = []
    all_labels = []

    for batch_idx, batch_data in tqdm(enumerate(dev_loader), total=len(dev_loader), desc=f'Testing epoch {epoch}... (Experiment {args.experiment_idx})'):

        with torch.no_grad():
            all_batch_scores = []
            input_cue, options_candidates, label_associations, num_associations = batch_data
            for item_in_batch, label_in_batch, input_cue_in_batch in zip(options_candidates, label_associations, input_cue):
                batch_scores = []

                inflated_input_cue = []
                option_input = []

                for option, label in zip(item_in_batch, label_in_batch):
                    if label.item() != -1:
                        inflated_input_cue.append(input_cue_in_batch.unsqueeze(0))
                        option_input.append(option.unsqueeze(0).unsqueeze(0))

                out = model(torch.cat(option_input), torch.cat(inflated_input_cue)).squeeze()
                batch_scores.append(out)
                all_batch_scores.append(torch.stack(batch_scores))

        y = label_associations.squeeze().to(device)

        if args.debug:
            if batch_idx > 2:
                break
        # loss = loss_fn(out, y)
        accuracy, predictions, labels = calculate_accuracy_test(all_batch_scores, y, num_associations)
        # epoch_dev_losses.append(loss.item())
        epoch_dev_accuracy += accuracy
        all_predictions += predictions
        all_labels += labels

    return epoch_dev_losses, epoch_dev_accuracy, all_predictions, all_labels


def calculate_accuracy_test(pred_batch, label_batch, num_associations_batch):

    batch_jaccard = []
    batch_preds = []
    batch_labels = []
    for pred, label, num_associations in zip(pred_batch, label_batch, num_associations_batch):

        label = label.cpu()
        pred = pred.cpu()

        top_k_preds_ind = np.argpartition(pred.numpy(), -num_associations)[0][-num_associations:]
        real_label = label.numpy()[np.where(label.numpy() != -1)]
        labels_indices = np.where(real_label == 1)[0]
        union = set(top_k_preds_ind).union(set(labels_indices))
        intersection = set(top_k_preds_ind).intersection(set(labels_indices))
        jaccard = len(intersection) / len(union)
        batch_jaccard.append(jaccard)
        batch_preds.append(top_k_preds_ind)
        batch_labels.append(labels_indices)

    return batch_jaccard, batch_preds, batch_labels


def dump_test_info(args, model_dir_path, all_losses, all_test_accuracy, test_df, epoch):
    test_losses_mean = {i: np.mean(v) for i, v in enumerate(all_losses['test'])}
    test_accuracy_mean = {i: np.mean(v) for i, v in enumerate(all_test_accuracy)}
    test_info = pd.concat(
        [pd.Series(test_losses_mean, name='test loss'), pd.Series(test_accuracy_mean, name='test accuracy')], axis=1)
    trainable_test_accuracy = round(test_info.iloc[0]['test accuracy'] * 100 , 2)
    out_p = os.path.join(model_dir_path, f'epoch_{epoch}_test')
    if args.result_suffix != '':
        out_p += "_" + args.result_suffix
    all_losses_out_p = out_p + '_all_losses_test.pickle'
    out_p_test_df = out_p + "_test_df.csv"
    out_p += ".csv"
    test_clip_zeroshot_jaccard = round(test_df['clip_vit_32_jaccard'].mean() * 100 , 2)
    print(f"test_clip_zeroshot_jaccard (# {len(test_df)} items): {test_clip_zeroshot_jaccard}")
    test_info['test_clip_zeroshot_jaccard'] = test_clip_zeroshot_jaccard
    test_info.to_csv(out_p)
    test_df.to_csv(out_p_test_df)
    all_losses_and_acc_d = {'all_losses': all_losses, 'all_test_accuracy': all_test_accuracy}
    with open(all_losses_out_p, 'wb') as f:
        pickle.dump(all_losses_and_acc_d, f)
    print(f'Dumping losses {len(test_info)} to {all_losses_out_p}')
    print(test_info)
    print(f'Dumping df {len(test_info)} to {out_p}, and {len(test_df)} to {out_p_test_df}')
    results_zeroshot_vs_trainable = {'zero-shot': test_clip_zeroshot_jaccard, 'trainable': trainable_test_accuracy}
    return results_zeroshot_vs_trainable


def main(args):
    splits = get_gvlab_data(args)
    backend_model = BackendModel(args.model_backend_type)
    baseline_model = BaselineModel(backend_model).to(device)
    print(f"Checking baseline model cuda: {next(baseline_model.parameters()).is_cuda}")

    if args.test_only:
        print(f'test_only - going to test')
        test(backend_model, baseline_model, splits)
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()
        if args.test_model is False:
            train(backend_model, baseline_model, splits, loss_fn)
        else:
            train(backend_model, baseline_model, splits, loss_fn)
            args.test_model = True
            results_zeroshot_vs_trainable = test(backend_model, baseline_model, splits)
            return results_zeroshot_vs_trainable
    return f"Finished experiment {args.experiment_idx}"



def train(backend_model, baseline_model, splits, loss_fn):
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=args.lr)
    train_dataset = Loader(splits['train'], backend_model)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    dev_dataset = Loader(splits['dev'], backend_model, is_train=False)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
    model_dir_path = get_experiment_dir(args)
    print(f"model_dir_path: {model_dir_path}")
    splits_path = os.path.join(model_dir_path, 'splits')
    if not os.path.exists(splits_path):
        os.mkdir(splits_path)
    for split in splits:
        if split == 'train':
            train_df = pd.DataFrame(splits['train'])
            print(f"Writing train df (# {len(train_df)} items)")
            train_df.to_csv(os.path.join(splits_path, f"train.csv"))
        else:
            splits[split].to_csv(os.path.join(splits_path, f"{split}.csv"))
    print(f"Wrote splits to {splits_path}")

    train_loop(args=args, model=baseline_model, optimizer=optimizer, train_loader=train_loader, dev_loader=dev_loader, loss_fn=loss_fn,
               n_epoch=args.n_epochs, model_dir_path=model_dir_path)


def train_loop(args, model, optimizer, train_loader, dev_loader, loss_fn, n_epoch, model_dir_path):
    """
    Parameters
    ----------
    args : (argparse.Namespace) arguments
    model :(nn.Module) The baseline model
    optimizer :(torch.optim.Optimizer)
    train_loader :(DataLoader) for the train set
    loss_fn : Loss function
    n_epoch :(int) epoch number
    """
    all_losses = {TRAIN: [], DEV: []}
    all_dev_accuracy = []

    for epoch in tqdm(range(n_epoch), desc=f'Training (Experiment {args.experiment_idx})'):
        epoch_train_losses = train_epoch(loss_fn, model, optimizer, train_loader, epoch)
        epoch_dev_losses, epoch_dev_accuracy, _, _ = test_epoch(model, dev_loader, epoch)
        all_losses[TRAIN].append(epoch_train_losses)
        all_losses[DEV].append(epoch_dev_losses)
        all_dev_accuracy.append(epoch_dev_accuracy)

        dev_accuracy_list = dump_train_info(args, model_dir_path, all_losses, all_dev_accuracy, epoch=epoch)
        save_model(model_dir_path, epoch, model)


def train_epoch(loss_fn, model, optimizer, train_loader, epoch):
    """
    Runs training on a single epoch
    Parameters
    ----------
    loss_fn : Loss function
    model : (nn.Module) The baseline model
    optimizer :(torch.optim.Optimizer)
    train_loader : (DataLoader)
    epoch : (int) epoch number
    Returns
    -------
    The epoch losses
    """
    model.train()
    epoch_train_losses = []

    with tqdm(enumerate(train_loader), total=len(train_loader)) as epochs:
        epochs.set_description(f'Training epoch {epoch}, split: {args.split} (Experiment {args.experiment_idx})')

        for batch_idx, batch_data in epochs:

            # Forward pass
            input_img, input_text, label = batch_data
            label = label.to(device)
            out = model(input_img, input_text).squeeze()

            y = label.squeeze()
            optimizer.zero_grad()

            # Compute Loss
            loss = loss_fn(out.double(), y.double())
            epoch_train_losses.append(loss.item())
            # Backward pass
            loss.backward()
            optimizer.step()

            if args.debug:
                if batch_idx > 2:
                    break

    return epoch_train_losses

if __name__ == '__main__':
    args = get_args()
    all_experiment_results = []
    for experiment_idx in range(args.num_experiments):
        setattr(args, 'experiment_idx', experiment_idx)
        if args.debug:
            setattr(args, 'n_epochs', 1)
        results_zeroshot_vs_trainable = main(args)
        all_experiment_results.append(results_zeroshot_vs_trainable)
        all_experiment_results_df = pd.DataFrame(all_experiment_results)
        print(f"*** experiment_idx : {experiment_idx} ***")
        print(all_experiment_results_df)
    all_experiment_results_df = pd.DataFrame(all_experiment_results)
    print(f"all_experiment_results_df:")
    print(all_experiment_results_df)
    print(f"MEAN")
    print(all_experiment_results_df.mean())
    print('STD')
    print(all_experiment_results_df.std())
