import argparse
import os
import sys
import pandas as pd

os.chdir(sys.path[0])


def process_dataset(json_path, save_file):
    print('#### Read the dataset...')
    if json_path.endswith('gz'):
        df = pd.read_json(json_path, lines=True, compression='gzip')
    else:
        df = pd.read_json(json_path, lines=True)
    df = df[['reviewerID', 'asin', 'overall']]
    df.to_csv(save_file, index=False, header=False)
    print(f'#### Saved {save_file}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', default='./Digital_Music_5.json')
    parser.add_argument('--save_file', dest='save_file', default='./music_5.csv')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)  # 文件夹不存在则创建
    process_dataset(args.data_path, args.save_file)
