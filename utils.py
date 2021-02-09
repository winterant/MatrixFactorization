import random
import time
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import DataLoader


def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())


def split_dataset(df: pd.DataFrame, test_num_per_star=1000):
    """
    :param df: 数据集
    :param test_num_per_star: 从数据集中，每个评分值中，抽取的样本数量
    :return: 训练集、测试集
    """
    stars = defaultdict(list)
    for idx, r in df['rating'].items():
        stars[r].append(idx)
    test_idx_list = list()
    for k in stars.keys():
        test_idx = random.sample(stars[k], min(len(stars[k]), test_num_per_star))
        test_idx_list.extend(test_idx)
    return df.drop(index=test_idx_list), df.loc[test_idx_list]


class MFDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.user_id = torch.LongTensor(df['userID'].to_list())
        self.item_id = torch.LongTensor(df['itemID'].to_list())
        self.rating = torch.Tensor(df['rating'].to_list())

    def __getitem__(self, idx):
        return self.user_id[idx], self.item_id[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]


def evaluate_mse(trained_model, dataloader, device):
    mse, sample_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            user_id, item_id, ratings = [i.to(device) for i in batch]
            predict = trained_model(user_id, item_id)
            mse += torch.nn.functional.mse_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mse / sample_count  # dataloader上的均方误差


def evaluate_precision(trained_model, dataloader, device):
    correct, sample_count = 0, 0
    yes = [0] * 6
    tot = [0] * 6
    with torch.no_grad():
        for batch in dataloader:
            user_id, item_id, ratings = [i.to(device) for i in batch]
            predict = trained_model(user_id, item_id)
            correct += ((predict - ratings).abs() <= 0.5).sum().item()
            for i in range(len(predict)):
                pred = int(ratings[i].item())
                tot[pred] += 1
                if (predict[i] - pred).abs() <= 0.5:
                    yes[pred] += 1
            sample_count += len(ratings)
    return correct / sample_count, [i / max(1, j) for i, j in zip(yes, tot)]


def evaluate_top_n(trained_model, device, test_data: pd.DataFrame, batch_size, candidate_items, topN=5):
    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, user_id_list, item_id_list):
            self.user_id = torch.LongTensor(user_id_list)
            self.item_id = torch.LongTensor(item_id_list)

        def __getitem__(self, idx):
            return self.user_id[idx], self.item_id[idx]

        def __len__(self):
            return self.user_id.shape[0]

    user_list, item_list = [], []
    for uid, row in test_data[['userID', 'itemID']][test_data['rating'] >= 4].groupby('userID'):
        curr_items = set(candidate_items).union(set(row['itemID']))
        user_list.extend([uid] * len(curr_items))
        item_list.extend(list(curr_items))
    dlr = DataLoader(TestDataset(user_list, item_list), batch_size=batch_size)

    top_n_list = defaultdict(list)
    with torch.no_grad():
        for batch in dlr:
            user_id, item_id = [i.to(device) for i in batch]
            predict = trained_model(user_id, item_id)
            for uid, iid, p in zip(user_id, item_id, predict):
                if len(top_n_list[uid]) < topN:
                    top_n_list[uid].append([p, iid])
                elif p > top_n_list[uid][-1][0]:
                    top_n_list[uid][-1] = (p, iid)
                top_n_list[uid].sort(reverse=True)

    pred_pos, real_pos = 0, 0
    for uid, row in test_data[test_data['rating'] >= 4].groupby('userID'):
        real_items = set(row['itemID'])
        pred_pos += len(real_items.intersection(top_n_list[uid]))
        real_pos += len(real_items)
    recall = pred_pos / real_pos
    precision = pred_pos / (len(top_n_list) * topN)
    return recall, precision
