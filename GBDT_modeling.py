import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import h5py
import os
from data_clean import pre_process,get_agg

#------------------------------定义评估标准---------------------------
def map5eval(preds,dtrain):
    actual = dtrain.get_label()
    predicted = preds.argsort(axis=1)[:-np.arange(5)]
    metric = 0
    for i in range(5):
        metric += np.sum(actual==predicted[:i])/(i+1)
    metric /= actual.shape[0]

    return 'map5',-metric

#------------------------------对模型进行训练-----------------------------------
clf = xgb.XGBClassifier(objective='multi:softmax',max_depth=5,n_estimators=300,learning_rate=0.01,nthread=4,subsample=0.7,colsample_bytree=0.7,min_child_weight=3,silent=False)
destinations = pd.read_csv('input/destinations.csv')
result = pd.read_csv('input/sample_result.csv')
agg1 = pd.read_csv('output/srch_dest_hc_hm_agg.csv')

if os.path.exists('rows_complete.txt'):
    with open('rows_complete.txt','r') as f:
        skipsize = int(f.readline())
else:
    skipsize = 0

skip = 0 if skipsize==0 else range(1,skipsize)
tchunksize = 1000000
print('%d rows will be skipped and next %d rows will be used for training' % (skipsize, tchunksize))
train = pd.read_csv('input/train.csv', parse_dates=['date_time', 'srch_ci', 'srch_co'], skiprows=skip, nrows=tchunksize)
train = train[train.is_booking==1]
train = pd.merge(train, destinations, how='left', on='srch_destination_id')
train = pd.merge(train, agg1, how='left', on=['srch_destination_id','hotel_country','hotel_market'])
pre_process(train)
y = train.hotel_cluster
train.drop(['cnt', 'hotel_cluster', 'is_booking'], axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(train, y, stratify=y, test_size=0.2)
clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric=map5eval, eval_set=[(X_train, y_train),(X_test, y_test)])

#-----------------------------对测试数据进行预测-----------------------------------
count = 0
chunksize = 10000
preds = np.empty((result.shape[0],clf.n_classes_))
reader = pd.read_csv('input/test.csv', parse_dates=['date_time', 'srch_ci', 'srch_co'], chunksize=chunksize)
for chunk in reader:
    chunk = pd.merge(chunk, destinations, how='left', on='srch_destination_id')
    chunk = pd.merge(chunk, agg1, how='left', on=['srch_destination_id', 'hotel_country', 'hotel_market'])
    chunk.drop(['id'], axis=1, inplace=True)
    pre_process(chunk)

    pred = clf.predict_proba(chunk)
    preds[count:(count + chunk.shape[0]), :] = pred
    count = count + chunksize
    print('%d rows completed' % count)

del clf
del agg1
if os.path.exists('output/probs/allpreds_xgb.h5'):
    with h5py.File('output/probs/allpreds_xgb.h5', 'r+') as hf:
        print('reading in and combining probabilities')
        predshf = hf['preds']
        preds += predshf.value
        print('writing latest probabilities to file')
        predshf[...] = preds
else:
    with h5py.File('../output/probs/allpreds_xgb.h5', 'w') as hf:
        print('writing latest probabilities to file')
        hf.create_dataset('preds', data=preds)

print('generating submission')
col_ind = np.argsort(-preds, axis=1)[:,:5]
hc = [' '.join(row.astype(str)) for row in col_ind]

sub = pd.DataFrame(data=hc, index=result.id)
sub.reset_index(inplace=True)
sub.columns = result.columns
sub.to_csv('output/pred_sub.csv', index=False)


skipsize += tchunksize
with open('rows_complete.txt', 'w') as f:
    f.write(str(skipsize))