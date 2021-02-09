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


def evaluate_mse(trained_model, dataloader):
    mse, sample_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            user_id, item_id, ratings = [i.to(next(trained_model.parameters()).device) for i in batch]
            predict = trained_model(user_id, item_id)
            mse += torch.nn.functional.mse_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mse / sample_count  # dataloader上的均方误差


def evaluate_precision(trained_model, dataloader):
    correct, sample_count = 0, 0
    yes = [0] * 6
    tot = [0] * 6
    with torch.no_grad():
        for batch in dataloader:
            user_id, item_id, ratings = [i.to(next(trained_model.parameters()).device) for i in batch]
            predict = trained_model(user_id, item_id)
            correct += ((predict - ratings).abs() <= 0.5).sum().item()
            for i in range(len(predict)):
                pred = int(ratings[i].item())
                tot[pred] += 1
                if (predict[i] - pred).abs() <= 0.5:
                    yes[pred] += 1
            sample_count += len(ratings)
    return correct / sample_count, [i / max(1, j) for i, j in zip(yes, tot)]


def evaluate_top_n(trained_model, dataloader: DataLoader, candidate_items, random_candi=0, topN=5):
    """
    top-N推荐系统评估
    :param trained_model: 训练好的模型
    :param dataloader: 测试集
    :param candidate_items: item候选集；list类型
    :param random_candi: 默认0表示整个item候选集；大于0时表示从item候选集中为每个user随机抽取的item数量
    :param topN: top-N推荐系统中的N
    :return: 召回率，精确率
    """
    # 真实购买记录（>=4分才算正样本），即测试集
    real_data = []
    for user_id, item_id, rating in dataloader.dataset:
        if rating >= 4:
            real_data.append([int(user_id), int(item_id)])
    real_data = pd.DataFrame(real_data, columns=['userID', 'itemID'])

    # 为每个user提供item候选列表，并构造为样本，用以预测top-N列表
    predict_data = []
    for uid, row in real_data.groupby('userID'):
        if random_candi > 0:
            user_candi_items = set(random.sample(candidate_items, k=random_candi)).union(set(row['itemID']))
        else:
            user_candi_items = set(candidate_items).union(set(row['itemID']))
        predict_data.extend([[uid, iid, 0] for iid in user_candi_items])
    predict_data = pd.DataFrame(predict_data, columns=['userID', 'itemID', 'rating'])  # 'rating' is useless
    dlr = DataLoader(MFDataset(predict_data), batch_size=dataloader.batch_size)

    # 开始为每个用户预测top-N列表
    top_n_list = defaultdict(list)
    with torch.no_grad():
        for batch in dlr:
            user_id, item_id, _ = [i.to(next(trained_model.parameters()).device) for i in batch]
            predict = trained_model(user_id, item_id)
            for user_id, item_id, pred in zip(user_id, item_id, predict):
                uid, iid, p = int(user_id), int(item_id), float(pred)
                if len(top_n_list[uid]) < topN:
                    top_n_list[uid].append([iid, p])
                elif p > top_n_list[uid][-1][0]:
                    top_n_list[uid][-1] = (iid, p)
                top_n_list[uid].sort(key=lambda x: -x[1])

    # 计算召回率，精确率
    pred_pos, real_pos = 0, 0
    for uid, row in real_data.groupby('userID'):
        real_items = set(row['itemID'])
        pred_pos += len(real_items.intersection([iid for iid, _ in top_n_list[uid]]))
        real_pos += len(real_items)
    recall = pred_pos / real_pos
    precision = pred_pos / (len(top_n_list) * topN)
    return recall, precision
