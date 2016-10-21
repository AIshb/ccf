#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

from config.filepath import *

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)
user = pd.read_csv(USER)
shop = pd.read_csv(SHOP)
trace = pd.read_csv(TRACE)

def get_nearest_neighbors_shop():
    nn = NearestNeighbors(n_jobs=-1)
    nn.fit(shop[['LONGITUDE', 'LATITUDE']])
    return nn

def generate_dataset(nn, k_shop_feature, k_classes, is_train=True):
    if is_train:
        df = pd.merge(train, user, on='USERID')
    else:
        df = pd.merge(test, user, on='USERID')
    df = pd.merge(df, trace, left_on=['USERID', 'ARRIVAL_TIME'], right_on=['USERID', 'BEGIN_TIME'])
    df.drop(['BEGIN_TIME'], axis=1, inplace=True)

    dist, idx_feature = nn.kneighbors(df[['STARTLONGTITUDE', 'STARTLATITUDE']], k_shop_feature)
    idx_classes = nn.kneighbors(df[['STARTLONGTITUDE', 'STARTLATITUDE']], k_classes, return_distance=False)

    n_classes = shop['CLASSIFICATION'].unique()
    n_classes.sort()
    classes_to_idx = {}
    for i, name in enumerate(n_classes):
        classes_to_idx[name] = i

    result = []
    for i in range(len(df)):
        # dist lon lat classification
        row = dist[i].tolist()
        t = shop.loc[idx_feature[i]][['LONGITUDE', 'LATITUDE', 'CLASSIFICATION']]
        row += t.values.T.flatten().tolist()
        
        # 5 features about user
        # lon lat duration
        row += df.loc[i][['INCOME', 'ENTERTAINMENT', 'BABY', 'GENDER', 'SHOPPING', 'STARTLONGTITUDE', 'STARTLATITUDE', 'DURATION']].values.flatten().tolist()

        # classes feature
        classes_list = [0] * len(n_classes)
        for j in idx_classes[i]:
            value = shop.loc[j]['CLASSIFICATION']
            classes_list[classes_to_idx[value]] += 1
        row += classes_list

        result.append(row)

    columns = ['{}_{}'.format(i, j) for i in ('dist', 'lon', 'lat', 'classification') for j in range(k_shop_feature)] \
            + ['INCOME', 'ENTERTAINMENT', 'BABY', 'GENDER', 'SHOPPING', 'STARTLONGTITUDE', 'STARTLATITUDE', 'DURATION'] \
            + ['n_classes_{}'.format(i) for i in n_classes]

    result = pd.DataFrame(result, columns=columns)
    if is_train:
        result['SHOPID'] = df['SHOPID'].copy()
        return result
    else:
        pred = df[['USERID', 'ARRIVAL_TIME']]
        return result, pred

    return result

def main():
    nn = get_nearest_neighbors_shop()
    train_set = generate_dataset(nn, 5, 30) 
    test_set, pred = generate_dataset(nn, 5, 30, False)

    train_set.to_csv('train_set', index=False)
    test_set.to_csv('test_set', index=False)

    train_set = train_set.values
    np.random.shuffle(train_set)

    clf = RandomForestClassifier(n_estimators=30, class_weight='balanced', n_jobs=-1)
    clf.fit(train_set[:, :-1], train_set[:, -1])

    pred.insert(1, 'SHOPID', clf.predict(test_set))
    pred[['USERID', 'SHOPID']] = pred[['USERID', 'SHOPID']].astype(int)
    pred.to_csv('output/clf_rf_2.csv', index=False)


if __name__ == '__main__':
    main()

