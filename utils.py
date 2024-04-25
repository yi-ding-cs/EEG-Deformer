from torch.utils.data import Dataset
import torch
import numpy as np
import os.path as osp
import pickle
import random
from torch.utils.data import DataLoader
import csv
import os
import re


class EEGDataset(Dataset):
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor
    def __init__(self, x, y):
        self.x = x
        self.y = y

        assert self.x.size(0) == self.y.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


def load_data_per_subject(load_path, sub):
    sub_code = 'sub' + str(sub) + '.pkl'
    path = osp.join(load_path, sub_code)
    with open(path, 'rb') as file:
        dataset = pickle.load(file)
    data = dataset['data']
    label = dataset['label']
    return data, label


def get_channel_info(load_path, graph_type):
    path_info = osp.join(load_path, 'dataset_info.pkl')
    with open(path_info, 'rb') as file:
        dataset_info = pickle.load(file)
    graph_idx = dataset_info[graph_type]
    original_order = dataset_info['BL']
    input_subgraph = False
    for item in graph_idx:
        if isinstance(item, list):
            input_subgraph = True
    idx_new = []
    num_chan_local_graph = []
    if not input_subgraph:
        for chan in graph_idx:
            idx_new.append(original_order.index(chan))
    else:
        for i in range(len(graph_idx)):
            num_chan_local_graph.append(len(graph_idx[i]))
            for chan in graph_idx[i]:
                idx_new.append(original_order.index(chan))
    return idx_new, num_chan_local_graph


def load_data(load_path, load_idx, keep_subject=False, concat=True):
    data, label = [], []
    for i, idx in enumerate(load_idx):
        data_per_sub, label_per_sub = load_data_per_subject(load_path=load_path, sub=idx)
        if keep_subject:
            data.append(data_per_sub)
            label.append(label_per_sub)
            assert concat is not True, "Please set concat False is keep_subject is True"
        else:
            data.extend(data_per_sub)
            label.extend(label_per_sub)
    if concat:
        data = np.concatenate(data)  # --> seg, chan, data
        label = np.concatenate(label)

    return data, label


def get_validation_set(train_idx, val_rate, shuffle):
    if shuffle:
        random.shuffle(train_idx)
    train = train_idx[:int(len(train_idx)*(1-val_rate))]
    val = train_idx[int(len(train_idx)*(1-val_rate)):]
    return train, val


def normalize(train, val, test):
    # input should be seg, chan, time/f
    # data: sample x channel x data
    for channel in range(train.shape[1]):
        mean = np.mean(train[:, channel, :])
        std = np.std(train[:, channel, :])
        train[:, channel, :] = (train[:, channel, :] - mean) / std
        val[:, channel, :] = (val[:, channel, :] - mean) / std
        test[:, channel, :] = (test[:, channel, :] - mean) / std
    return train, val, test


def numpy_to_torch(data, label):
    data = torch.from_numpy(data).float()
    label = torch.from_numpy(label).long()
    return data, label


def get_dataloader(data, label, batch_size, shuffle=True):
    # load the data
    dataset = EEGDataset(data, label)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    return loader


def prepare_data_for_training(data, label, idx, batch_size, shuffle):
    # reorder the data segment, chan, datapoint
    data = data[:, idx, :]
    # change to torch tensor
    data, label = numpy_to_torch(data=data, label=label)
    # prepare dataloader
    data_loader = get_dataloader(data=data, label=label, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def get_task_chunk(subjects, step):
    return np.array_split(subjects, len(subjects) // step)


def log2txt(text_file, content):
    file = open(text_file, 'a')
    file.write(str(content) + '\n')
    file.close()


def log2csv(csv_file, content):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row (column names)
        writer.writerow(["Metric", "Value"])

        # Write the data from the dictionary
        for key, value in content.items():
            writer.writerow([key, value])

    print(f"Data has been logged to {csv_file}")


def get_checkpoints(path):
    all_files = os.listdir(path)
    ckpt_files = [file for file in all_files if file.endswith(".ckpt")]
    return ckpt_files


def get_epoch_from_ckpt(ckpt_file):
    epoch_match = re.search(r'epoch=(\d+)', ckpt_file)
    epoch_number = int(epoch_match.group(1))
    return epoch_number


class LabelSmoothing(torch.nn.Module):
    """NLL loss with label smoothing.
       refer to: https://github.com/NVIDIA/DeepLearningExamples/blob/8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/ConvNets/image_classification/smoothing.py#L18
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


