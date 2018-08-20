# 数据清理

import pandas as pd
import numpy as np
import datetime
import os

#-------------------数据清洗--------------------

def pre_process(data):
    try:
        data.loc[data.srch_ci.str.endswith('00'), 'srch_ci'] = '2015-12-31'
        data['srch_ci'] = data.srch_ci.astype(np.datetime64)
        data.loc[data.date_time.str.endswith('00'), 'date_time'] = '2015-12-31'
        data['date_time'] = data.date_time.astype(np.datetime64)
    except:
        pass
    data.fillna(0, inplace=True)
    data['srch_duration'] = data.srch_co - data.srch_ci
    data['srch_duration'] = data['srch_duration'].apply(lambda td: td / np.timedelta64(1, 'D'))
    data['time_to_ci'] = data.srch_ci - data.date_time
    data['time_to_ci'] = data['time_to_ci'].apply(lambda td: td / np.timedelta64(1, 'D'))
    data['ci_month'] = data['srch_ci'].apply(lambda dt: dt.month)
    data['ci_day'] = data['srch_ci'].apply(lambda dt: dt.day)
    # data['ci_year'] = data['srch_ci'].apply(lambda dt: dt.year)
    data['bk_month'] = data['date_time'].apply(lambda dt: dt.month)
    data['bk_day'] = data['date_time'].apply(lambda dt: dt.day)
    # data['bk_year'] = data['date_time'].apply(lambda dt: dt.year)
    data['bk_hour'] = data['date_time'].apply(lambda dt: dt.hour)
    data.drop(['date_time', 'user_id', 'srch_ci', 'srch_co'], axis=1, inplace=True)

def get_agg():
    reader = pd.read_csv('input/train.csv', parse_dates=['date_time', 'srch_ci', 'srch_co'], chunksize=200000)
    pieces = [
        chunk.groupby(['srch_destination_id', 'hotel_country', 'hotel_market', 'hotel_cluster'])['is_booking'].agg(
            ['sum', 'count']) for chunk in reader]
    agg = pd.concat(pieces).groupby(level=[0, 1, 2, 3]).sum()
    del pieces
    agg.dropna(inplace=True)
    agg['sum_and_cnt'] = 0.85 * agg['sum'] + 0.15 * agg['count']
    agg = agg.groupby(level=[0, 1, 2]).apply(lambda x: x.astype(float) / x.sum())
    agg.reset_index(inplace=True)
    agg1 = agg.pivot_table(index=['srch_destination_id', 'hotel_country', 'hotel_market'], columns='hotel_cluster',
                           values='sum_and_cnt').reset_index()
    agg1.to_csv('output/srch_dest_hc_hm_agg.csv', index=False)
    del agg

get_agg()



