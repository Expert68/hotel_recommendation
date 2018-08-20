from sklearn.ensemble import RandomForestClassifier
from data_clean import pre_process
import pandas as pd
import numpy as np
import h5py
import os

#------------------------------对模型进行训练-----------------------------------
clf = RandomForestClassifier(n_estimators=10,n_jobs=6,warm_start=True,verbose=1)
count = 0
chunksize = 200000
destinations = pd.read_csv('input/destinations.csv')
result = pd.read_csv('input/sample_result.csv')
agg1 = pd.read_csv('output/srch_dest_hc_hm_agg.csv')
reader = pd.read_csv('input/train.csv', parse_dates=['date_time', 'srch_ci', 'srch_co'], chunksize=chunksize)

for chunk in reader:
    try:
        chunk = chunk[chunk.is_booking == 1]
        chunk = pd.merge(chunk, destinations, how='left', on='srch_destination_id')
        chunk = pd.merge(chunk, agg1, how='left', on=['srch_destination_id', 'hotel_country', 'hotel_market'])
        pre_process(chunk)
        y = chunk.hotel_cluster
        chunk.drop(['cnt', 'hotel_cluster', 'is_booking'], axis=1, inplace=True)

        if len(y.unique()) == 100: #一共100个旅馆，一定要凑齐所有的旅馆才能进行训练
            clf.set_params(n_estimators=clf.n_estimators+1)  #每一个chunk加一棵树
            clf.fit(chunk,y)

        count += chunksize
        print('%d rows completed' %count)
        if(count/chunksize) == 300:
            break
    except Exception as e:
        print('出现错误：%s' %str(e))

print(clf.n_classes_)
#-----------------------------对测试数据进行预测-----------------------------------
count = 0
chunksize = 10000
preds = np.empty((result.shape[0],clf.n_classes_))
reader = pd.read_csv('input/test.csv',parse_dates=['date_time', 'srch_ci', 'srch_co'], chunksize=chunksize)
for chunk in reader:
    chunk = pd.merge(chunk, destinations, how='left', on='srch_destination_id')
    chunk = pd.merge(chunk, agg1, how='left', on=['srch_destination_id', 'hotel_country', 'hotel_market'])
    chunk.drop(['id'], axis=1, inplace=True)
    pre_process(chunk)

    pred = clf.predict_proba(chunk)
    preds[count:(count + chunk.shape[0]), :] = pred
    count = count + chunksize
    print('%d rows completed' % count)

#-----------------------------将结果写入到文件中-------------------------------

if os.path.exists('output/probs/allpreds.h5'):
    with h5py.File('output/probs/allpreds.h5', 'r+') as hf:
            print('reading in and combining probabilities')
            predslatesthf = hf['preds_latest']
            preds += predslatesthf.value
            print('writing latest probabilities to file')
            predslatesthf[...] = preds
            print(predslatesthf[...])
else:
    with h5py.File('output/probs/allpreds.h5', 'w') as hf:
        print('writing latest probabilities to file')
        hf.create_dataset('preds_latest', data=preds)



col_ind = np.argsort(-preds, axis=1)[:,:5]
hc = [' '.join(row.astype(str)) for row in col_ind]

sub = pd.DataFrame(data=hc, index=result.id)
sub.reset_index(inplace=True)
sub.columns = result.columns
sub.to_csv('output/pred_result.csv', index=False)

