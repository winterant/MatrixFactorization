import time
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import Config
from utils import date, evaluate_mse, MFDataset, evaluate_top_n, evaluate_precision


class FunkSVD(nn.Module):

    def __init__(self, M, N, K=10):
        super().__init__()
        self.user_emb = nn.Parameter(torch.randn(M, K))
        self.user_bias = nn.Parameter(torch.randn(M))  # 偏置
        self.item_emb = nn.Parameter(torch.randn(N, K))
        self.item_bias = nn.Parameter(torch.randn(N))
        self.bias = nn.Parameter(torch.zeros(1))  # 全局偏置

    def forward(self, user_id, item_id):
        pred = self.user_emb[user_id] * self.item_emb[item_id]
        pred = pred.sum(dim=-1) + self.user_bias[user_id] + self.item_bias[item_id] + self.bias
        return pred


def train(train_dataloader, valid_dataloader, model, config, model_path):
    print(f'{date()}## Start the training!')
    train_mse = evaluate_mse(model, train_dataloader, config.device)
    valid_mse = evaluate_mse(model, valid_dataloader, config.device)
    print(f'{date()}#### Initial train mse {train_mse:.6f}, validation mse {valid_mse:.6f}')
    start_time = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, config.learning_rate_decay)

    best_loss = 100
    for epoch in range(config.train_epochs):
        model.train()  # 将模型设置为训练状态
        total_loss, total_samples = 0, 0
        for batch in train_dataloader:
            user_id, item_id, ratings = [i.to(config.device) for i in batch]
            predict = model(user_id, item_id)
            loss = F.mse_loss(predict, ratings, reduction='mean')
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(predict)
            total_samples += len(predict)

        lr_sch.step()
        model.eval()  # 停止训练状态
        valid_mse = evaluate_mse(model, valid_dataloader, config.device)
        train_loss = total_loss / total_samples
        print(f"{date()}#### Epoch {epoch:3d}; train mse {train_loss:.6f}; validation mse {valid_mse:.6f}")

        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)

    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')


def main():
    config = Config()
    print(config)

    df = pd.read_csv(config.dataset_file, usecols=[0, 1, 2])
    df.columns = ['userID', 'itemID', 'rating']  # Rename above columns for convenience
    # map user(or item) to number
    df['userID'] = df.groupby(df['userID']).ngroup()
    df['itemID'] = df.groupby(df['itemID']).ngroup()
    user_count = df['userID'].value_counts().count()  # 用户数量
    item_count = df['itemID'].value_counts().count()  # item数量
    print(f"{date()}## Dataset contains {df.shape[0]} records, {user_count} users and {item_count} items.")

    # 划分训练集、验证集、测试集；
    # $方式1：按8:1:1的比例随机划分；
    train_data, valid_data = train_test_split(df, test_size=1 - 0.8, random_state=3)
    valid_data, test_data = train_test_split(valid_data, test_size=0.5, random_state=4)
    # $方式2：对于每个评分值，抽取固定数量的样本作为验证集/测试集；
    # train_data, valid_data = split_dataset(df, test_num_per_star=2000)
    # valid_data, test_data = split_dataset(valid_data, test_num_per_star=1000)

    train_dataset = MFDataset(train_data)
    valid_dataset = MFDataset(valid_data)
    test_dataset = MFDataset(test_data)
    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size)

    model = FunkSVD(user_count, item_count, config.hidden_K).to(config.device)
    train(train_dlr, valid_dlr, model, config, config.saved_model)

    print(f'{date()}## Start the testing!')
    trained_model = torch.load(config.saved_model)
    test_loss = evaluate_mse(trained_model, test_dlr, next(model.parameters()).device)
    print(f'{date()}## Test for Rating Prediction: mse is {test_loss:.6f}')

    overall_precision, each_precision = evaluate_precision(trained_model, test_dlr, next(model.parameters()).device)
    print(f'{date()}## Test for Rating Prediction: Overall Precision is {overall_precision:.4f};'
          f'Precision of every star is {[int(i * 1e4) / 1e4 for i in each_precision]}')

    recall, precision = evaluate_top_n(torch.load(config.saved_model), config.device, test_data,
                                       batch_size=config.batch_size,
                                       candidate_items=df['itemID'].unique().tolist(), topN=5)
    print(f'{date()}## Test for Top-N Recommender: Recall@{5} is {recall:.4f}; Precision is {precision:.4f}')


if __name__ == '__main__':
    main()
