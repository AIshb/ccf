#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def main():
    trace = pd.read_csv('datasets/USER_TRACE.csv')
    test = pd.read_csv('datasets/TEST2.csv')
    shop = pd.read_csv('datasets/SHOP_PROFILE.csv')
    
    df = pd.merge(test, trace, left_on=['USERID', 'ARRIVAL_TIME'], right_on=['USERID', 'BEGIN_TIME'])

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(shop[['LONGITUDE', 'LATITUDE']])
    idx = nn.kneighbors(df[['STARTLONGTITUDE', 'STARTLATITUDE']], return_distance=False)
    idx = [x[0] for x in idx]

    result = df[['USERID', 'ARRIVAL_TIME']]
    result.insert(1, 'SHOPID', shop.iloc[idx, 4].reset_index(drop=True))
    result[['USERID', 'SHOPID']] = result[['USERID', 'SHOPID']].astype(int)
    result.to_csv('output/baseline.csv', index=False)

    print(result)


if __name__ == '__main__':
    main()

