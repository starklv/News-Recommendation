import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


# all_click_df 指的是训练集
# sample_user_nums 采样作为验证集的用户数量
def trn_val_split(all_click_df, sample_user_nums):
    # replace = True 表示可以重复抽样，反之不可以
    sample_user_ids = np.random.choice(all_user_ids, size=sample_user_nums, replace=False)

    click_val = all_click[all_click['user_id'].isin(sample_user_ids)]
    click_trn = all_click[~all_click['user_id'].isin(sample_user_ids)]

    # 将验证集中的最后一次点击给抽取出来作为答案
    click_val = click_val.sort_values(['user_id', "click_timestamp"])
    val_ans = click_timestamp.groupby("user_id").tail(1)

    click_val = click_val.groupby("user_id").apply(lambda x: x[:-1]).reset_index(drop=True)

    # 去除val_ans中某些用户只有一个点击数据的情况，如果该用户只有一次点击情况，会给模型验证带来困难
    val_ans = val_ans[val_ans['user_id'].isin(click_val['user_id'].unqiue())]
    click_val = click_val[click_val['user_id'].isin(val_ans['user_id'].unique())]

    return click_trn, click_val, click_ans


# 将召回列表转换成df的形式
def recall_dict_2_df(recall_list_dict):
    df_row_list = []  # [user,  item, score]
    for user, recall_list in tqdm(recall_list_dict.items()):
        for item, score in recall_list:
            df_row_list.append([user, item, score])

    col_names = ['user_id', 'sim_item', 'score']
    recall_list_df = pd.DataFrame(df_row_list, columns=col_names)

    return recall_list_df


# 召回数据打标签， label 为1代表是正样本，指的是召回列表中是最后一次召回的样本，
# label 为0代表是负样本， 代表并不是最后一次召回的样本
def get_rank_label_df(recall_list_df, label_df, is_test=False):
    # 测试集没有标签，但为了和后面代码统一， 这里直接给一个负数代替
    if is_test:
        recall_list_df['label'] = -1
        return recall_list_df

    label_df = label_df.rename(columns={"click_article_id": "sim_item"})
    recall_list_df_ = recall_list_df.merge(label_df[['user_id', 'sim_item', 'click_timestamp']],
                                           how="left", on=["user_id", "sim_item"])
    recall_list_df_["label"] = recall_list_df_["click_timestamp"].apply(lambda x: 0.0 if np.isnan(x) else 1.0)
    del recall_list_df_['click_timestamp']

    return recall_list_df_


# 负采样函数， 这里可以控制负采样的比例，这里给了一个默认的值
def neg_sample_recall_data(recall_items_df, sample_rate=0.001):
    pos_data = recall_items_df[recall_items_df["label"] == 1]
    neg_data = recall_items_df[recall_items_df["label"] == 0]

    # 分组采样函数
    def neg_sample_func(group_df):
        neg_num = len(group_df)
        # 保证最少一个
        sample_num = max(int(neg_num * sample_rate), 1)
        # 保证最多不超过5个， 这里可以根据实际情况选择
        sample_num = min(sample_num, 5)
        return group_df.sample(n=sample_num, replace=False)

    # 对用户进行负采样， 保证所有用户都在采样后的数据中
    neg_data_user_sample = neg_data.groupby('user_id', group_keys=False).apply(neg_sample_func)
    # 对文章进行负采样， 保证所有文章都在采样后的数据中
    neg_data_item_sample = neg_data.groupby('sim_item', group_keys=False).apply(neg_sample_func)

    # 将上述两种情况下的采样数据合并
    neg_data_new = neg_data_user_sample.append(neg_data_item_sample)
    # 由于上述两个操作分开，可能将两个相同的数据重复选择， 所以需要对合并后的数据进行去重
    neg_data_new = neg_data_new.sort_values(['user_id', 'score']).drop_duplicates(['user_id', 'sim_item'],
                                                                                  keep="last")

    data_new = pd.concat([pos_data, neg_data], ignore_index=True)

    return data_new


def get_user_recall_item_label_df(click_trn_hist, click_val_hist, click_tst_hist,
                                 click_trn_last, click_val_last, recall_list_df):
    # 获取训练数据的召回列表
    trn_user_items_df = recall_list_df[recall_list_df['user_id'].isin(click_trn)]


# 将最终召回的df数据转换成字典形式做排序特征
def make_tuple_func(group_df):
    row_data = []
    for name, row_df in group_df.iterrows():
        row_data.append((row_df['click_article_id'], row_df['click_environment']))

    return row_data

trn_user_item_label_tuples = trn_user_item_label_df.groupby('user_id').\
                                apply(make_tuple_func).reset_index()
trn_user_item_label_tuples_dict = dict(zip(trn_user_item_label_tuples['user_id'],
                                          trn_user_item_label_tuples[0]))


def hist_func(user_df):
    if len(user_df) == 1:
        return user_df
    else:
        return user_df[:-1]


# 获取文章id对应的基本属性，保存成字典的形式，方便后面召回阶段，冷启动阶段直接使用
def get_item_info_dict(item_info_df):
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    item_info_df['created_at_ts'] = item_info_df[['created_at_ts']].apply(max_min_scaler)

    item_type_dict = dict(zip(item_info_df['click_article_id'], item_info_df['category_id']))
    item_words_dict = dict(zip(item_info_df['click_article_id'], item_info_df['words_count']))
    item_created_time_dict = dict(zip(item_info_df['click_article_id'], item_info_df['created_at_ts']))

    return item_type_dict, item_words_dict, item_created_time_dict


def create_feature(users_id, recall_list, click_hist_df, articles_info,
                   articles_emb, user_emb=None, N=1):
    """
        基于用户的历史行为做相关特征
        ：param user_id： 用户id
        ：param recall_list：对于每个用户召回的候选文章列表
        ：param click_hist_df：用户的历史点击信息
        ：param articles_info：文章信息
        ：param articles_emb： 文章的embedding向量，这个可以用item_content_emb, item_w2v_emb, item_youtube_emb
        ：param N：最近的N次点击， 由于testA日志里面很多用户只存在一次历史点击，为了不产生空值，默认是1
    """
    # 建立一个二维列表保存结果，后面要转成DataFrame
    all_user_feas = []
    i = 0
    for user_id in tqdm(users_id):
        # 该用户的最后N次点击
        hist_user_items = click_hist_df[click_hist_df['user_id'] == user_id]['click_article_id'][-N:]

        # 遍历该用户的召回列表
        for rank, (article_id, score, label) in enumerate(recall_list[user_id]):
            # 该文章建立时间， 字数
            a_create_time = item_words_dict[article_id]
            a_words_count = item_created_time_dict[article_id]
            single_usr_fea = [user_id, article_id]
            # 计算与最后点击商品的相似度的和， 最大值和最小值， 均值
            sim_fea = []
            time_fea = []
            word_fea = []
            # 遍历用户的最后N次点击文章
            for hist_item in hist_user_items:
                b_created_time = item_created_time_dict[article_id]
                b_words_count = item_words_dict[article_id]

                sim_fea.append(np.dot(articles_emb[hist_item], articles_emb[article_id]))
                time_fea.append(abs(a_created_time - b_created_time))
                word_fea.append(abs(a_words_count - b_words_count))

            single_user_fea.extend(sim_fea)  # 相似性特征
            single_user_fea.extend(time_fea)  # 时间差特征
            single_user_fea.extend(word_fea)  # 字数差特征
            # 相似性统计特征
            single_user_fea.extend([max(sim_fea), min(sim_fea), sum(sim_fea), sum(sim_fea) / len(sim_fea)])

            single_user_fea.extend([score, rank, label])
            all_user_feas.append(single_usr_fea)

        # 定义列名
        id_cols = ['user_id', 'click_article_id']
        sim_cols = ['sim' + str(i) for i in range(N)]
        time_cols = ['time_diff' + str(i) for i in range(N)]
        word_cols = ['word_diff' + str(i) for i in range(N)]
        sat_cols = ['sim_max', 'sim_min', 'sim_sum', 'sim_mean']
        user_score_rank_label = ['score', 'rank', 'label']
        cols = id_cols + sim_cols + time_cols + word_cols + sat_cols + user_score_rank_label

        # 转成DataFrame
        df = pd.DataFrame(all_user_feas, columns=cols)

        return df

    def active_level(all_data, cols):
        """
        制作区分用户活跃度的特征
        ：param all_data：数据集
        ：param cols： 用到的特征列
        """
        data = all_data[cols]
        data.sort_values(['user_id', 'click_timestamp'], in_place=True)
        user_act = pd.DataFrame(data.groupby('user_id', as_index=False)[['click_article_id', 'click_timestamp']] \
                                .agg({'click_article_id': np.size, "click_timestamp": list}).values,
                                columns=['user_id', 'click_size', 'click_timestamp'])

        # 计算时间间隔的均值
        def time_diff_mean(l):
            if len(l) == 1:
                return 100
            else:
                return np.mean([j - i for i, j in list(zip(i[:-1], l[1:]))])

        # 点击次数取倒数
        user_act['click_size'] = 1 / user_act['click_size']

        def max_min_scaler(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x))

        # 两者归一化
        user_act['click_size'] = user_act[['click_size']].apply(max_min_scaler)
        user_act['time_diff_mean'] = user_act[['time_diff_mean']].apply(max_min_scaler)
        user_act['active_level'] = user_act['click_size'] + user_act['time_diff_mean']

        user_act['user_id'] = user_act['user_id'].astype(int)
        del user_act['click_timestamp']

        return user_act


def device(all_data, cols):
    """
    制作用户的设备特征
    ：param all_data：数据集
    ：param cols：用到的特征列
    """
    user_device = all_data[cols]

    # 用众数来表示每个用户的设备信息
    user_device_info = user_device_info.groupby('user_id').agg(lambda x: x.value_counts().index[0]) \
        .reset_index()

    return user_device_info

# 设备特征
device_cols = ['user_id', 'click_environment', 'click_deviceGroup', 'click_os', 'click_country', 'click_region',
                   'click_referrer_type']


def user_time_hob_fea(all_data, cols):
    """
    制作用户的时间习惯特征
    ：param all_data: 数据集
    ：param cols： 用到的特征列
    """
    user_time_hob_info = all_data[cols]

    # 先把时间戳进行归一化
    mm = MinMaxScaler()
    user_time_hob_info['click_timestamp'] = mm.fit_transform(user_time_hob_info[['click_timestamp']])
    user_time_hob_info['created_at_ts'] = mm.fit_transform(user_time_hob_info[['created_at_ts']])

    user_time_hob_info = user_time_hob_info.groupby('user_id').agg('mean').reset_index()
    user_time_hob_info.renames('click_timestamp': 'user_time_hob1', 'created_at_ts': 'user_time_hob2'}, inplace = True)

    return user_time_hob_info


def user_cat_hot_fea(all_data, cols):
    """
    用户的主题爱好
    ：param all_data：数据集
    ：param cols：用到的特征列
    """
    user_catgory_hob_info = all_data[cols]
    user_category_hob_info = user_category_hob_info.groupby('user_id').agg(list).reset_index()

    user_cat_hot_info = pd.DataFrame()
    user_cat_hot_info['user_id'] = user_category_hob_info['user_id']
    user_cat_hot_info['cate_list'] = user_cat_category_hob_info['category_id']

    return user_cat_hot_info

trn_user_item_feats_df['is_cat_hab'] = trn_user_item_feats_df.apply(
                    lambda x: 1 if  x.category_id in set(x.cate_list) else 0, axis = 1)





