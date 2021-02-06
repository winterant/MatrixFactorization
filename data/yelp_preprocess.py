import argparse
import jsonlines
import os
import sys
import pandas as pd

os.chdir(sys.path[0])


# 由于yelp的json文件太大（5.8GB），只能分块读取
def process_dataset(json_path, save_file):
    print('#### Read the dataset...')
    data = []
    with open(json_path, 'r', encoding='UTF-8') as f:
        for item in jsonlines.Reader(f):
            line = [item["user_id"], item["business_id"], item["stars"]]
            data.append(line)
    df = pd.DataFrame(data=data, columns=['user_id', 'business_id', 'stars'])
    df['user_id'] = df.groupby(df['user_id']).ngroup()
    df['business_id'] = df.groupby(df['business_id']).ngroup()
    df.to_csv(save_file, index=False, header=False)
    print(f'#### Saved {save_file}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path',
                        default='F:/dataset/yelp2020/yelp_dataset-2/yelp_academic_dataset_review.json')
    parser.add_argument('--save_file', dest='save_file', default='./yelp_ratings.csv')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)  # 文件夹不存在则创建
    process_dataset(args.data_path, args.save_file)
