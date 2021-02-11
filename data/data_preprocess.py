import argparse
import gzip

import jsonlines
import os
import sys
import pandas as pd

os.chdir(sys.path[0])


# 由于json文件太大（如5GB），只能分块读取
def process_dataset(json_path, col_name, save_file):
    print('#### Read the dataset...')
    data = []
    if json_path.endswith('.gz'):
        f = gzip.open(json_path, 'rb')
    else:
        f = open(json_path, 'r', encoding='UTF-8')
    for item in jsonlines.Reader(f):
        line = [item[k] for k in col_name]
        data.append(line)
    f.close()

    df = pd.DataFrame(data=data, columns=['userID', 'itemID', 'rating'])
    df['userID'] = df.groupby(df['userID']).ngroup()
    df['itemID'] = df.groupby(df['itemID']).ngroup()
    df.to_csv(save_file, index=False, header=False)
    print(f'#### Saved {save_file}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', default='Digital_Music_5.json')
    parser.add_argument('--data_source', dest='data_source', default='amazon')
    parser.add_argument('--save_file', dest='save_file', default='./amazon_ratings.csv')
    args = parser.parse_args()

    if args.data_source == 'amazon':
        col_name = ['reviewerID', 'asin', 'overall']
    else:
        col_name = ['user_id', 'business_id', 'stars']
    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    process_dataset(args.data_path, col_name, args.save_file)
