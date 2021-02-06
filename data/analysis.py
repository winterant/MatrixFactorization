import argparse
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', default='./movie-ratings.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.data_path, usecols=[0, 1, 2])
    df.columns = ['userID', 'itemID', 'rating']  # Rename above columns for convenience

    vc = df.value_counts(['rating'], sort=False)
    idx = [i[0] for i in vc.index]

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(10, 10))
    rects_fig = plt.bar(idx, vc.values.tolist(), color='#9999ff', width=0.3)

    # 添加数据标签 就是矩形上面的数值
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height + 0.01 * height, '%.0f' % height, ha='center',
                     va='bottom', fontsize=20, color='red')
            rect.set_edgecolor('white')

    add_labels(rects_fig)
    plt.title(args.data_path+'评分分布柱状图')
    plt.xlabel('评分')
    plt.ylabel('数量')
    plt.show()
