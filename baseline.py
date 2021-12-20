import pandas as pd
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import math
import pickle


data_path = "./data/"
save_path = "./data/temp_results/"
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)


# debug模式: 从训练集中划分出一部分数据来测试
def get_all_click_sample(data_path, sample_nums=10000):
    """
        训练集中采样一部分数据调试
        data_path：原数据的存储路径
        sample_nums：采样数目
    """
    all_click = pd.read_csv(os.path.join(data_path, "train_click_log.csv"))
    all_user_ids = all_click.user_id.unique()

    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False)
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)]

    all_click = all_click.drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'])
    return all_click


# 读取点击数据，这里分为线上和线下，如果是为了获取线上提交结果应该将测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集
def get_all_click_df(data_path, offline=True):
    if offline:
        all_click = pd.read_csv(os.path.join(data_path, "train_click_log.csv"))
    else:
        train_click = pd.read_csv(os.path.join(data_path, "train_click_log.csv"))
        test_click = pd.read_csv(os.path.join(data_path, "testA_click_log.csv"))

        all_click = train_click.append(test_click)

    all_click = all_click.drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'])
    return all_click

# 全量训练
all_click_df = get_all_click_df(data_path, offline=False)
test_click_df = get_all_click_sample(data_path)


# 根据点击时间获取用户的点击文章序列 {user1: [(item1, time1), (item2, time2)]}
def get_user_item_time(click_df):
    click_df = click_df.sort_values('click_timestamp')

    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))

    user_item_time_df = click_df.groupby('user_id')[['click_article_id', 'click_timestamp']]. \
        apply(lambda x: make_item_time_pair(x)).reset_index().rename(columns={0: 'item_time_list'})

    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))

    return user_item_time_dict


# 获取近期点击最多的文章
def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click


def itemcf_sim(df):
    """
        文章与文章之间的相似性矩阵计算
        :param df:数据表
        :item_created_time_dict: 文章创建时间的字典
        return: 文章与文章的相似度矩阵
    """

    user_item_time_dict = get_user_item_time(df)

    # 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        for i, i_click_time in item_time_list:
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for j, j_click_time in item_time_list:
                if (i == j):
                    continue
                i2i_sim[i].setdefault(j, 0)
                i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1)

    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])

    # 将得到的相似性矩阵保存到本地
    with open(os.path.join(save_path, "itemcf_i2i_sim.pkl"), "wb") as fw:
        pickle.dump(i2i_sim_, fw)

    return i2i_sim_

i2i_sim_ = itemcf_sim(all_click_df)

def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num,
                         item_topk_click):
    """
        基于文章协同过滤的召回
        ：param user_id： 用户id
        ：param user_item_time_dict：字典，根据点击时间获取用户的点击文章序列  {user:[(item1,time1),
        (item2, user2)]...}
        ：param i2i_sim：字典，文章相似度矩阵
        ：param sim_item_topk：整数，选择与当前文章最相似的前k篇文章
        ：param recall_item_num：整数，最后的召回文章数量
        ：param item_topk_click：列表，点击次数最多的文章列表，用户召回补全
        return：召回的文章列表{item1:score1, item2:score2....}
    """

    # 获取用户历史交互的文章
    user_hist_items = user_item_time_dict[user_id]
    user_hist_items_ = {item_id for item_id, _ in user_hist_items}

    item_rank = defaultdict(int)
    for loc, (i, click_time) in enumerate(user_hist_items):
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_items_:
                continue

            item_rank[j] = item_rank.get(j, 0) + wij

    # 如果不足10个， 用热门商品补全
    if len(item_rank) < recall_item_num:
        print("support")
        for i, item in enumerate(item_topk_click):
            if item in item_rank.keys():  # 填充的item应该不在原来的列表中
                continue
            item_rank[item] = - i - 100  # 随便给个负数补充排序
            if len(item_rank) == recall_item_num:
                break

    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return item_rank

# 定义
user_recall_items_dict = defaultdict(dict)

# 获取用户 - 文章 - 点击时间的字典
user_item_time_dict = get_user_item_time(all_click_df)

# 获取文章相似度
with open(os.path.join(save_path, 'itemcf_i2i_sim.pkl'), 'rb') as fr:
    i2i_sim = pickle.load(fr)

# 相似文章的数量
sim_item_topk = 10

# 召回文章数量
recall_item_num = 10

# 用户热度补全
item_topk_click = get_item_topk_click(all_click_df, k=50)

for user_id in tqdm(all_click_df['user_id'].unique()):
    user_recall_items_dict[user] = item_based_recommend(user_id, user_item_time_dict, i2i_sim,
                                                        sim_item_topk, recall_item_num, item_topk_click)

# aa