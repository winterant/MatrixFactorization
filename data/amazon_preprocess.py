import os
import sys
import pandas as pd

os.chdir(sys.path[0])


def process_dataset(json_path):
    print('#### Read the dataset...')
    if json_path.endswith('gz'):
        df = pd.read_json(json_path, lines=True, compression='gzip')
    else:
        df = pd.read_json(json_path, lines=True)
    df = df[['reviewerID', 'asin', 'overall']]
    df.to_csv(json_path+'.csv', index=False, header=False)
    print(f'#### Saved {json_path}.csv!')


if __name__ == '__main__':
    process_dataset('Digital_Music_5.json')
    # process_dataset('reviews_Clothing_Shoes_and_Jewelry_5.json.gz')
