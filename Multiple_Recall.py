import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle
from collections import defaultdict
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss
from tensorflow.keras.models import Model
from datetime import datetime


data_path = "./data/"
save_path = "./data/temp_results/"


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


# 读取文章的基本属性
def get_item_info_df(data_path):
    item_info_df = pd.read_csv(os.path.join(data_path, "articles.csv"))

    # 为了方便与训练集中的click_article_id 拼接，需要把article_id修改成click_article_id
    item_info_df = item_info_df.rename(columns={"article_id": "click_article_id"})

    return item_info_df

# 读取文章的Embedding数据
def get_item_emb_dict(data_path):
    item_emb_df = pd.read_csv(os.path.join(data_path, "articles_emb.csv"))

    item_emb_cols = [x for x in item_emb_df.columns if "emb" in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols])
    # 进行归一化
    item_emb_dict = dict(zip(item_emb_df['article_id'], item_emb_np))
    with open(os.path.join(save_path, "item_content_emb.pkl"), "wb") as fw:
        pickle.dump(item_emb_dict, fw)

    return item_emb_dict

max_min_scaler = lambda x:(x-np.min(x))/(np.max(x) - np.min(x))

# 全量训练集
all_click_df = get_all_click_df(data_path, offline=False)

# 对时间戳进行归一化，用于在关联规则的时候计算权重
all_click_df['click_timestamp'] = all_click_df[['click_timestamp']].apply(max_min_scaler)

#item_info_df = get_item_info_df(data_path)

#item_emb_dict = get_item_emb_dict(data_path)

# 根据点击时间获取用户的点击文章序列 {user1: [(item1, time1), (item2, time2)]}
def get_user_item_time(click_df):
    click_df = click_df.sort_values('click_timestamp')

    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))

    user_item_time_df = click_df.groupby('user_id')[['click_article_id', 'click_timestamp']]. \
        apply(lambda x: make_item_time_pair(x)).reset_index().rename(columns={0: 'item_time_list'})

    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))

    return user_item_time_dict


# 根据点击时间获取用户的点击文章序列 {item1: [(user1, time1), (user2, time2)]}
def get_item_user_time_dict(click_df):
    def make_user_time_pair(df):
        return list(zip(df['user_id'], df['click_timestamp']))

    click_df = click_df.sort_values('click_timestamp')
    item_user_time_df = click_df.groupby('click_article_id')[['user_id', 'click_timestamp']]. \
        apply(lambda x: make_item_time_pair(x)).reset_index().rename(columns={0: 'item_time_list'})

    item_user_time_dict = dict(zip(item_user_time_df['click_article_id'], item_user_time_df['item_time_list']))

    return item_user_time_dict


# 获取当前数据的历史点击和最后一次点击
def get_hist_and_last_click(all_click):
    all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
    click_last_df = all_click.groupby('user_id').tail(1)

    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]

    click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)

    return click_hist_df, click_last_df


def get_user_hist_item_info_dict(all_click):
    # 获取user_id对应的用户历史点击文章类型的集合字典
    user_hist_item_typs = all_click.groupby('user_id')['category_id'].agg(set).reset_index()
    user_hist_item_typs_dict = dict(zip(user_hist_item_typs['user_id'], user_hist_item_typs['category_id']))

    # 获取user_id对应的用户点击文章的集合
    user_hist_item_ids_dict = all_click.groupby('user_id')['click_article_id'].agg(set).reset_index()
    user_hist_item_ids_dict = dict(zip(user_hist_item_ids_dict['user_id'], user_hist_item_ids_dict['click_article_id']))

    # 获取user_id对应的用户历史点击的文章的平均字数字典
    user_hist_item_words = all_click.groupby('user_id')['words_count'].agg('mean').reset_index()
    user_hist_item_words_dict = dict(zip(user_hist_item_words['user_id'], user_hist_item_words['words_count']))

    # 获取user_id对应的用户最后一次点击的文章的创建时间
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    all_click_ = all_click.sort_values('click_timestamp')
    all_click['created_at_ts'] = all_click[['created_at_ts']].apply(max_min_scaler)
    user_last_item_created_time = all_click.groupby('user_id').tail(1)

    user_last_item_created_time_dict = dict(zip(user_last_item_created_time['user_id'], \
                                                user_last_item_created_time['created_at_ts']))

    return user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created


# 获取近期点击最多的文章
def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click

# 获取文章的属性信息，保存成字典的形式方便查询
item_type_dict, item_words_dict, item_created_time_dict = get_item_info_dict(item_info_df)


# 依次评估召回的前10,20,30,40,50个文章中的击中率
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=10):
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'],
                                    trn_last_click_df['click_article_id']))
    user_num = len(user_recall_items_dict)

    for k in range(10, topk + 1, 10):
        hit_num = 0
        for user, item_list in user_recall_items_dict.items():
            tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
            if last_click_item_dict[user] in set(tmp_recall_items):
                hit_num += 1

        hit_rate = round(hit_num * 1.0 / user_num, 5)
        print(' topk :{0}, hit_num :{1}, hit_rate :{2}, user_num :{3}'
              .format(k, hit_num, hit_rate, user_num))


def itemcf_sim(df, item_created_time_dict):
    """
        文章与文章之间的相似性矩阵计算
        ：param df：数据表
        ：item_created_time_dict： 文章创建时间的字典
        return ：文章与文章的相似性矩阵
        思路：基于物品的协同过滤
    """

    user_item_time_dict = get_user_item_time(df)

    # 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        for loc1, (i, i_click_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_click_time) in enumerate(item_time_list):
                if (i == j):
                    continue

                # 考虑文章的正向顺序点击和反向顺序点击
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                # 位置信息权重，其中的参数可以调节
                loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                # 点击时间权重，其中的参数可以调节
                click_time_weight = np.exp(0.7 ** np.abs(i_click_time - j_click_time))
                # 两篇文章创建时间的权重，其中的参数可以调节
                created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
                i2i_sim[i].setdefault(j, 0)
                # 考虑多种因素的权重计算最终的文章之间的相似度
                i2i_sim[i][j] += loc_weight * click_time_weight * created_time_weight / math.log(
                    1 + len(user_item_time_dict))

        i2i_sim_ = i2i_sim.copy()
        for i, related_items in i2i_sim.items():
            for j, wij in related_items.items():
                i2i_sim[i][j] = wij / (math.sqrt(item_cnt[i] * item_cnt[j]))

                # 将得到的相似性矩阵保存到本地
        with open(os.path.join(save_path, 'itemcf_i2i_sim.pkl'), 'wb') as fw:
            pickle.dump(i2i_sim_, fw)

        return i2i_sim_


def get_user_activate_degree_dict(all_click_df):
    all_click_df_ = all_click_df.groupby('user_id')['click_article_id'].count().reset_index()

    all_click_df['click_article_id'] = all_click_df_[['click_article_id']].apply(max_min_scaler)
    user_activate_degree_dict = dict(zip(all_click_df_['user_id'], all_click_df_['click_article_id']))

    return user_activate_degree_dict


def usercf_sim(all_click_df, user_activate_degree_dict):
    """
        用户相似度计算
        ：param all_click_df：数据表
        ：param user_activate_degree_dict：用户活跃度的字典
        return 用户相似性矩阵
    """
    item_user_time_dict = get_item_user_time_dict(all_click_df)

    u2u_sim = {}
    user_cnt = defaultdict(int)
    for item, user_time_list in tqdm(item_user_time_dict):
        for u, click_time in user_time_list:
            user_cnt[u] += 1
            u2u_sim.setdefault(u, {})
            for v in user_time_list:
                u2u_sim[u].setdefault(v, 0)
                if u == v:
                    continue
                # 用户平均活跃度作为活跃度的权重，
                u2u_sim[u][v] += 1 / math.sqrt(len(user_time_list) + 1)

    u2u_sim_ = u2u_sim
    for u, related_users in u2u_sim.items():
        for v, wij in related_users.items():
            u2u_sim[u][v] = wij / math.sqrt(user_cnt[u] * user_cnt[v])

    # 将得到的

user_activate_degree_dict = get_user_activate_degree_dict(all_click_df)
u2u_sim = usercf_sim(all_click_df, user_activate_degree_dict)


# 向量检索相似度计算
# topk指的是每个item, faiss搜索后返回最相似的topk个item
def embedding_sim(click_df, item_emb_df, save_path, topk):
    """
        基于内容的文章embedding相似性矩阵计算
        ：param click_df：数据表
        ：param item_emb_df：文章的embedding
        ：param save_path：保存路径
        ：param topk：找最相似的topk篇
        return 文章相似性矩阵

        思路：对于每一篇文章， 基于embedding的相似性返回topk个与其最相似的文章，只不过文章数量太多，
        这里用了faiss进行加速
    """
    # 文章索引与文章id的字典映射
    item_idx_2_rawid_dict = dict(zip(item_emb_df.index, item_emb_df['article_id']))

    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols].values, dtype=np.float32)
    # 向量进行单位化
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    # 建立faiss索引
    item_index = faiss.IndexFlatIP(item_emb_np.shape[1])
    item_index.add(item_emb_np)
    # 相似度查询，给每个索引位置上的向量返回topk个item以及相似度
    sim, idx = item_index.search(item_emb_np, topk)  # 返回的是列表

    # 将向量检索的结果保存成原始id的对应关系
    item_sim_dict = defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(item_emb_np), sim, idx))):
        target_raw_id = item_idx_2_rawid_dict[target_idx]
        item_sim_dict.setdefault(target_raw_id, {})
        # 从1开始是为了去除商品本身，所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = rele_idx_list[rele_idx]
            item_sim_dict[target_raw_id].setdefault(rele_raw_id, 0)
            item_sim_dict[target_raw_id][rele_raw_id] += sim_value

    # 保存i2i相似度矩阵
    with open(os.path.join(save_path, 'emb_i2i_sim.pkl'), 'wb') as fw:
        pickle.dump(item_sim_dict, fw)

    return item_sim_dict


# negsample指的是通过滑窗构建样本的时候，负样本的数量
def gen_data_set(data, negsample=5):
    data.sort_values("click_timestamp", inplace=True)
    item_ids = data['click_article_id'].unique()

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['click_article_id'].tolist()

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))  # 用户没看过的文章里面选择负样本
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=False)
            # 对于每个正样本，选择n个负样本

        # 长度只有一个的时候， 需要把这条数据也放到训练集中，不然的话最终学到的embedding就会有缺失
        if len(pos_list) == 1:
            train_set.append((reviewerID, [pos_list[0]], pos_list[0], 1, len(pos_list)))
            test_set.append((reviewerID, [pos_list[0]], pos_list[0], 1, len(pos_list)))

        # 滑窗构造正负样本
        for i in range(1, len(pos_list)):
            hist = po_list[:i]

            if i != len(pos_list) - 1:
                # 正样本 [user_id, hist_item, pos_item, label, len(hist_item)]
                train_set.append((reviewerID, hist, pos_list[i], 1, len(hist)))
                for negi in range(negsample):
                    # 负样本 [user_id, his_item, neg_item, label, len(his_item)]
                    train_set.append((reviewerID, hist, neg_list[i * negsample + negi], 0, len(hist)))
            else:
                # 将最长的那一个序列长度作为测试数据
                test_set.append((reviewerID, hist, pos_list[i], 1, len(hist)))

        random.shuffle(train_set)
        random.shffle(test_set)

        return train_set, test_set


# 将输入的数据进行padding, 使得序列特征的长度都一致
def gen_model_input(train_set, user_profile, seq_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding="post", truncating="post", value=0)
    train_model_input = {"user_id": train_uid, "click_article_id": train_iid, "hist_article_id": train_seq_pad,
                         "hist_len": train_hist_len}

    return train_model_input, train_label


def youtubednn_u2i_dict(data, topk=20):
    sparse_features = ["click_article_id", "user_id"]
    SEQ_LEN = 30  # 用户点击序列的长度，短的填充，长的截断

    user_profile_ = data[['user_id']].drop_duplicates('user_id')
    item_profile_ = data[['click_article_id']].drop_duplicates('user_id')

    # 类别编码
    features = ["click_article_id", "user_id"]
    feature_max_idx = {}

    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature])
        feature_max_idx[feature] = data[feature].max() + 1

    # 提取user和Item的画像，这里具体选择那些特征还需要进一步分析和考虑
    user_profile = data[['user_id']].drop_duplicates('user_id')
    item_profile = data[['click_article_id']].drop_duplicates('user_id')

    user_index_2_rawid = dict(zip(user_profile['user_id'], user_profile_['user_id']))
    item_index_2_rawid = dict(zip(item_profile['click_article_id'], item_profile_['click_article_id']))

    # 划分训练集和测试集
    # 由于深度学习需要的数据量通常是非常大的，所以为了保证召回的效果，往往会通过滑窗的形式扩充训练样本
    train_set, test_set = gen_data_set(data, 5)
    # 整理输入数据，具体的操作可以看上面的函数
    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

    # 确定Embedding的维度
    embedding_dim = 16

    # 将数据整理成模型可以直接输入的形式
    user_feature_columns = [SparseFeat(['user_id'], feature_max_id['user_id'], embedding_dim),
                            VarLenSparseFeat(SparseFeat('hist_article_id', feature_max_id['click_article_id'],
                                                        embedding_dim, embedding_name="click_article_id"), SEQ_LEN,
                                             "mean", "hist_len")]
    item_feature_columns = [SparseFeat('click_article_id', feature_max_idx['click_article_id'], embedding_dim)]

    # 模型定义
    # num_sampled：负采样时的样本数量
    model = YoutubeDNN(user_feature_columns, item_feature_columns, num_sampled=5,
                       user_dnn_hidden_units=(64, embedding_dim))
    # 模型编译
    model.compile(optimizer="adam", loss=sampledsoftmaxloss)

    # 模型训练，这里可以定义验证集的比例，如果设置为0的话是全量数据直接进行训练
    history = model.fit(train_model_input, train_label, batch_size=256, verbose=1, epochs=1, validation_split=0.0)

    # 训练完模型之后，提取训练的Embedding，包括user端和Item端
    test_user_model_input = test_model_input
    all_item_model_input = {"click_article_id": item_profile["click_article_id"].values}

    user_embedding_model = Model(inputs=model.user_input, output=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, output=model.item_embedding)

    # 保存当前的item_embedding 和 user_embedding 排序的时候可能能够用到，但是需要注意保存的时候和原始id对应
    user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    item_embs = user_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

    # embedding保存之前归一化一下
    user_embs = user_embs / np.linalg.norm(user_embs, axis=1, keepdims=True)
    item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)

    # embedding转换成字典的形式方便查询
    raw_user_id_emb_dict = {user_index_2_rawid[k]: \
                                v for k, v in zip(user_profile['user_id'], user_embs)}
    raw_item_id_emb_dict = {item_index_2_rawid[k]: \
                                v for k, v in zip(user_profile['click_article_id'], item_embs)}

    # 将Embedding保存到本地
    with open(os.path.join(save_path, "user_youtube_emb.pkl"), "wb") as fw:
        pickle.dump(raw_user_id_emb_dict, fw)
    with open(os.path.join(save_path, "item_youtube_emb.pkl"), "wb") as fw:
        pickle.dump(raw_item_id_emb_dict, fw)

    # faiss近邻搜索，通过user_embedding 搜索与其相似性最高的topk个item
    index = faiss.IndexFlatIP(embedding_dim)
    # 将item向量构建索引
    index.add(item_embs)
    # 通过user去查询最相似的topk个item
    sim, idx = index.search(np.ascontiguousarray(user_embs), topk)

    user_recall_item_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True) for k, v
                             in user_recall_item_emb_dict.items()}
    # 将召回的结果排序
    # 保存召回的结果
    with open(os.path.join(save_path, "youtube_u2i_dict.pkl"), "wb") as fw:
        pickle.dump(user_recall_items_dict, fw)

    return user_recall_items_dict


# 基于商品的召回i2i
def item_created_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num,
                           item_topk_click, item_created_time_dict, emb_i2i_sim):
    """
        基于文章协同过滤的召回
        ：param user_id：用户id
        ：param user_item_time_dict：字典，根据点击时间获取用户的点击文章序列 {user：[(item1, time1), ...]}
        ：param i2i_sim：字典，文章相似度矩阵
        ：param sim_item_topk：整数，选择与当前文章最相似的前k篇文章
        ：param item_topk_click：列表，点击次数最多的文章列表，用户召回补全
        ：param emb_i2i_sim：字典基于内容embedding算的文章相似度矩阵

        return：召回的文章列表[(item1, score1), ...]
    """
    # 获取用户历史交互的文章
    user_hist_items = user_item_time_dict[user_id]
    user_hist_items_ = {item_id for item_id, time in user_hist_items}

    item_rank = {}
    for loc, (i, click_time) in enumerate(user_item_time_dict[user_id].items()):
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_items_:
                continue

            # 文章创建时间差权重
            created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
            # 相似文章和历史点击文章序列中历史文章所在的位置权重
            loc_weight = (0.9 ** (len(user_hist_items) - loc)

                          content_weight = 1.0
                          if emb_i2i_sim.get(i, {}).get(j, None) is not None:
            content_weight += emb_i2i_sim[i][j]
            if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                content_weight += emb_i2i_sim[j][i]

            item_rank.setdefault(j, 0)
            item_rank[j] += created_time_weight * loc_weight * content_weight

            # 不足10个，用热门商品补全
            if len(item_rank) < recall_item_num:



            # 基于用户的召回u2i

                def user_based_recommend(user_id, user_item_time_dict, u2u_sim, sim_user_topk,
                                         recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim):
                    """
                        基于文章协同过滤的推荐
                        ：param user_id：用户id
                        ：param user_item_time_dict：字典，根据点击时间获取用户的点击文章序列 {}
                        ：param u2u_sim：字典，文章相似性矩阵
                        ：param sim_user_topk：整数， 选择与当前用户最相似的前k个账户
                        ：param recall_item_num：整数，最后的召回文章数量
                        ：param item_topk_click：列表，点击次数最多的文章列表，用户召回补全
                        ：param item_created_time_dict： 文章创建时间列表
                        ：param emb_i2i_sim：字典基于内容embedding算的文章相似度矩阵
                    """
                    # 历史交互
                    # [(item1, time1), (item2, time2), ...]
                    user_item_time_list = user_item_time_dict[user_id]
                    # 存在一个用户与某篇文章的多次交互，这里得去重
                    user_hist_items = set([item for item, time in user_item_time_list])

                    item_rank = {}
                    for sim_u, wuv in sorted(u2u_sim[user_id].items(), key=lambda x: x[1], reverse=True)[
                                      :sim_user_topk]:
                        for i, click_time in user_item_time_dict[sim_u]:
                            if i in user_hist_items:
                                continue
                            items_rank.setdefault(i, 0)

                            loc_weight = 1.0
                            content_weight = 1.0
                            created_time_weight = 1.0

                            # 当前文章与该用户看的历史文章进行一个权重交互
                            for loc, (j, click_time) in enumerate(user_item_time_list):
                                # 点击时的相对位置权重
                                loc_weight += 0.9 ** (len(user_item_time_list) - loc)
                                # 内容相似性权重
                                if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                                    content_weight += emb_i2i_sim[i][j]
                                if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                                    content_weight += emb_i2i_sim[j][i]

                                # 创建时间差权重
                                created_time_weight += np.exp(0.8 * np.abs(item_created_time_dict[i] -
                                                                           item_created_time_dict[j]))

                            item_rank[i] += loc_weight * content_weight * created_time_weight * wuv

                        # 热度补全
                        if len(items_rank) < recall_item_num:
                            for i, item in enumerate(item_topk_click):
                                # 填充的item应该不在原来列表中
                                if item in items_rank.items():
                                    continue
                                # 随便给个负数就行
                                items_rank[item] = -i - 100
                                if len(items_rank) == recall_item_num:
                                    break

                        items_rank = sorted(items_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_rank]

                        return items_rank


# 基于规则进行文章过滤
# 保留文章主题与用户历史浏览主题相似的文章
# 保留文章字数与用户历史浏览文章字数相差不大的文章
# 保留最后一次点击当天的文章
# 按照相似度返回最终的结果

def get_click_article_ids_set(all_click_df):
    return set(all_click_df['click_article_id'].values)


def cold_start_items(user_recall_items_dict, user_hist_item_types_dict, user_hist_item_words_dict,
                     user_last_item_created_time_dict, item_type_dict, item_words_dict,
                     item_created_time_dict, click_article_ids_set, recall_item_num):
    """
        ：param user_recall_items_dict：基于内容embedding相似性来召回的很多文章， 字典， {user1:[(item1,
        item2)]}
        ：param user_hist_item_types_dict：字典， 用户点击的文章的主题映射
        ：param user_hist_item_words_dict：字典， 用户点击的历史文章的平均字数
        ：param user_last_item_created_time_dict：字典，用户点击的历史文章创建时间映射
        ：param item_type_dict：字典，文章主题映射
        ：param item_created_time_dict：字典，文章字数映射
        ：param item_words_dict：字典，文章字数映射
        ：param click_article_ids_set：集合，用户点击过的文章，也就是日志里面出现过的文章
        ：param recall_item_num：召回文章的数量， 这个指的是没有出现在日志里面的文章数量
    """

    cold_start_user_items_dict = {}
    for user, item_list in tqdm(user_recall_item_dict.items()):
        cold_start_user_items_dict.setdefault(user, [])
        for item, score in item_list:
            # 获取历史文章信息
            hist_item_type_set = user_hist_item_types_dict[user]
            hist_mean_words = user_hist_item_words_dict[user]
            hist_last_item_created_time = user_last_item_created_time_dict[user]
            hist_last_item_created_time = datetime.fromtimestamp(hist_last_item_created_time)

            # 获取当前召回文章的信息
            curr_item_type = item_type_dict[item]
            curr_item_words = item_words_dict[item]
            curr_item_created_time = item_created_time_dict[item]
            curr_item_created_time = datetime.fromtimestamp(curr_item_created_time)

            # 首先，文章不能出现在用户的历史点击中， 然后根据文章主题， 文章单词数， 文章创建时间进行筛选
            if curr_item_type not in hist_type_set or \
                    item in click_article_ids_set or \
                    abs(curr_item_words - hist_mean_words) > 200 or \
                    abs((curr_item_created_time - hist_last_item_created_time).days > 90):
                continue

            # {user1:[(item1, score1), ...]}
            cold_start_user_items_dict[user].append((item, socre))

    # 需要控制以下冷启动召回的数量
    cold_start_user_items_dict = {k: sorted(v, key=lambda x: x[1], reverse=True)[:recall_item_num] \
                                  for k, v in cold_start_user_items_dict.items()}


def combine_recall_results(user_multi_recall_dict, weight_dict=None, topk=25):
    final_recall_items_dict = {}

    # 对每一种召回结果按照用户进行归一化， 方便后面多种召回结果， 相同用户的物品之间的权重相加
    def norm_user_recall_items_sim(sorted_item_list):
        # 如果冷启动中没有文章或者只有一篇文章，直接返回
        # 为了各个召回方式的score进行比较，需要先进行min_max归一化
        if len(sorted_item_list) < 2:
            return sorted_item_list

        min_sim = sorted_item_list[-1][1]
        max_sim = sorted_item_list[0][1]

        normed_sorted_item_list = []
        for item, score in sorted_item_list:
            if max_sim > 0:
                norm_score = 1.0 * (score - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 1.0
            else:
                norm_score = 0.0
            normed_sorted_list.append((item, norm_score))

        return normed_sorted_item_list

    print("多路召回合并....")
    for method, user_recall_items in tqdm(user_multi_recall_dict.items()):
        print(method + "...")
        # 在计算最终召回结果的时候，需要为每一种召回结果设置一个权重
        if weight_dict = None:
            recall_method_weight = 1
        else:
            recall_method_weight = weight_dict[method]

        # 进行归一化
        for user_id, sorted_item_list in user_recall_items.items():
            user_recall_items[user_id] = norm_user_recall_items_sim(sorted_item_list)

        for user_id, sorted_item_list in user_recall_items.items():

            final_recall_items_dict.setdefault(user_id, {})
            for item, score in sorted_item_list:
                final_recall_items_dict[user_id].setdefault(item, 0)
                final_recall_items_dict[user_id][item] += score * recall_method_weight

    final_recall_items_dict_rank = {k: sorted(v, key=lambda x: x[1], reverse=True)[:topk] for
                                    k, v in final_recall_items_dict.items()}



