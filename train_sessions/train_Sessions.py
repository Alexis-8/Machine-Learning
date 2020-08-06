# logit + bag of sites
# holdout 10% -- 91%

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix

train_df = pd.read_csv('train_sessions.csv', index_col = 'session_id')
test_df = pd.read_csv('test_sessions.csv', index_col='session_id')


times = ["time%s" % i for i in range(1,11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

train_df = train_df.sort_values(by = 'time1')

# будем применять bag of words (one hot encoding) для сайтов, не трогая пока время вообще

sites = ['site%s' % i for i in range(1, 11)]

train_df[sites] = train_df[sites].fillna(0).astype('int')
test_df[sites] = test_df[sites].fillna(0).astype('int')

with open('site_dic.pkl', 'rb') as input_file:
    site_dict = pickle.load(input_file)
# print(site_dict.keys())
# print(site_dict.values())


# dictionary of sites:

sites_dict_df = pd.DataFrame(list(site_dict.keys()),
                             index = site_dict.values(),
                             columns=['site'])

train_labels = train_df['target']

full_df = pd.concat([train_df.drop('target', axis=1), test_df])

idx_split = train_df.shape[0]

sites = ["site%d" % i for i in range(1,11)]


full_sites = full_df[sites]

# приведение к разреженному формату: 1 сессия: [индексы сайтов]

sites_flatten = full_sites.values.flatten()

full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0], sites_flatten, range(0, sites_flatten.shape[0] + 10, 10)))

train_data_sparse = full_sites_sparse[:idx_split]
test_data_sparse = full_sites_sparse[idx_split:]

# Данных много, отложим выборку, не будем делать кросс-валидацию
# измерим качество на отложенной выборке:

def get_auc_valid(data, labels, C = 1.0, ratio = 0.9, seed = 3):
    """C - коэф регуляризации логистической регрессии
    seed = random state"""
    train_len = int(ratio*data.shape[0])

    train_data = data[:train_len, :]
    valid_data = data[train_len:, :]

    train_labels = labels[:train_len]
    valid_labels = labels[train_len:]


    logit = LogisticRegression(C=C, n_jobs=-1, random_state=seed) # n_jobs - распараллеливание процессов -1 на все ядра
    logit.fit(train_data, train_labels)

    # представление в вероятностях для отложенной выборки:

    valid_pred = logit.predict_proba(valid_data)[:, 1]

    return roc_auc_score(valid_labels, valid_pred)


get_auc_valid(train_data_sparse, train_labels) # 0.915579876193892

# был прогноз на 90% данных обучающей выборки
# теперь сделаем прогноз на тестовой выборке test_data_sparse:

logit = LogisticRegression(n_jobs=-1, random_state=3)
logit.fit(train_data_sparse, train_labels)

test_pred = logit.predict_proba(test_data_sparse)[:, 1] # вероятности, что это Элис (второй столбец, т.е. вероятности 1)

# добавим еще временные признаки:

time = ["time%d" % i for i in range(1,11)]

# посмотрим только на начало: time1:

new_feat_train = pd.DataFrame(index=train_df.index) # placeholders
new_feat_test = pd.DataFrame(index=test_df.index)

new_feat_train['year_month'] = train_df['time1'].apply(lambda ts: 100*ts.year+ ts.month).head
new_feat_test['year_month'] = test_df['time1'].apply(lambda ts: 100*ts.year+ ts.month).head

# получили признак очень большого порядка, нужно маштабирование признаков:

scaler = StandardScaler() # из каждого столбца вычитаем среднее и делим на стандартное отколоение
scaler.fit(new_feat_train['year_month'])

# новый масштабированный признак:
#
# new_feat_train['year_month_scaled'] = scaler.transform(new_feat_train['year_month'].values.reshape(-1,1))
# new_feat_test['year_month_scaled'] = scaler.transform(new_feat_test['year_month'].values.reshape(-1,1))
#
# train_data_sparse_new = csr_matrix(hstack([train_data_sparse, new_feat_train['year_month_scaled'].values.reshape(-1,1)]))
#
# print(get_auc_valid(train_data_sparse_new,train_labels))
