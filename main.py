import time
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from config import Config


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


class CFDataset(Dataset):
    def __init__(self, df):
        self.user_id = torch.LongTensor(df['userID'].to_list())
        self.item_id = torch.LongTensor(df['itemID'].to_list())
        self.rating = torch.Tensor(df['rating'].to_list())

    def __getitem__(self, idx):
        return self.user_id[idx], self.item_id[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]


def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())


def predict_mse(trained_model, dataloader, device):
    mse, sample_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            user_id, item_id, ratings = [i.to(device) for i in batch]
            predict = trained_model(user_id, item_id)
            mse += torch.nn.functional.mse_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mse / sample_count  # dataloader上的均方误差


def train(train_dataloader, valid_dataloader, model, config, model_path):
    print(f'{date()}## Start the training!')
    train_mse = predict_mse(model, train_dataloader, config.device)
    valid_mse = predict_mse(model, valid_dataloader, config.device)
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
        valid_mse = predict_mse(model, valid_dataloader, config.device)
        train_loss = total_loss / total_samples
        print(f"{date()}#### Epoch {epoch:3d}; train mse {train_loss:.6f}; validation mse {valid_mse:.6f}")

        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)

    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')


def test(dataloader, model):
    print(f'{date()}## Start the testing!')
    start_time = time.perf_counter()
    test_loss = predict_mse(model, dataloader, next(model.parameters()).device)
    end_time = time.perf_counter()
    print(f"{date()}## Test end, test mse is {test_loss:.6f}, time used {end_time - start_time:.0f} seconds.")


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

    train_data, valid_data = train_test_split(df, test_size=1 - 0.8, random_state=3)  # split dataset including random
    valid_data, test_data = train_test_split(valid_data, test_size=0.5, random_state=4)
    train_dataset = CFDataset(train_data)
    valid_dataset = CFDataset(valid_data)
    test_dataset = CFDataset(test_data)
    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size)

    model = FunkSVD(user_count, item_count, config.hidden_K).to(config.device)
    train(train_dlr, valid_dlr, model, config, config.model_file)
    test(test_dlr, torch.load(config.model_file))


if __name__ == '__main__':
    main()
