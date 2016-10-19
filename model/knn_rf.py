#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

from config.filepath import *

def main():
    test = pd.read_csv(TEST)
    train = pd.read_csv(TRAIN)
    trace = pd.read_csv(TRACE)
    shop = pd.read_csv(SHOP)
    user = pd.read_csv(USER)
    knn = 20

    # get knn shop
    test_location = pd.merge(test, trace, left_on=['USERID', 'ARRIVAL_TIME'], right_on=['USERID', 'BEGIN_TIME'])
    nn = NearestNeighbors(n_neighbors=knn)
    nn.fit(shop[['LONGITUDE', 'LATITUDE']])
    idx = nn.kneighbors(test_location[['STARTLONGTITUDE', 'STARTLATITUDE']], return_distance=False)

    # generate train/test set
    df = pd.merge(train[['USERID', 'SHOPID']], user, on='USERID')
    df = pd.merge(df, shop[['ID', 'CLASSIFICATION']], left_on='SHOPID', right_on='ID')
    df = df.drop(['USERID', 'SHOPID', 'ID'], axis=1)
    train_x = df.drop(['CLASSIFICATION'], axis=1)
    train_y = df['CLASSIFICATION']
    print(train_x.shape, train_y.shape)
    
    # train a classifier
    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(train_x, train_y)

    # get test label
    test_user = pd.merge(test_location[['USERID', 'ARRIVAL_TIME']], user, on='USERID')
    test_label = clf.predict(test_user.drop(['USERID', 'ARRIVAL_TIME'], axis=1))
    print(test_label)

    # get shop id
    result = test_user[['USERID', 'ARRIVAL_TIME']]
    shop_id = []
    for i in range(len(result)):
        is_find = False
        for j in range(knn):
            if test_label[i] == shop['CLASSIFICATION'][idx[i][j]]:
                shop_id.append(int(shop['ID'][idx[i][j]]))
                is_find = True
                break
        if not is_find:
            shop_id.append('')

    print(shop_id)

    result.insert(1, 'SHOPID', shop_id)
    result['USERID'] = result['USERID'].astype(int)
    result.to_csv('output/knn_rf.csv', index=False)


if __name__ == '__main__':
    main()

