import pandas as pd
import datetime
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import SGDClassifier
import numpy as np
import h5py
import pickle
import os

cat_col = ['user_id', 'user_location_city',
           'srch_destination_id', 'srch_destination_type_id', 'hotel_continent',
           'hotel_country', 'hotel_market']

num_col = ['is_mobile', 'is_package']

def bin_time(t):
    if t < 0:
        x = 0
    elif t < 2:
        x = 1
    elif t < 7:
        x = 2
    elif t < 30:
        x = 3
    else:
        x = 4


def pre_process(data):
    try:
        data.loc[data.srch_ci.str.endswith('00'), 'srch_ci'] = '2015-12-31'
        data['srch_ci'] = data.srch_ci.astype(np.datetime64)
        data.loc[data.date_time.str.endswith('00'), 'date_time'] = '2015-12-31'
        data['date_time'] = data.date_time.astype(np.datetime64)
    except:
        pass
    data.fillna(0, inplace=True)

    data['ci_month'] = data['srch_ci'].apply(lambda dt: dt.month)
    data['season_dest'] = 'season_dest' + data.ci_month.map(str) + '*' + data.srch_destination_id.map(str)
    data['season_dest'] = data['season_dest'].map(hash)
    data['time_to_ci'] = data.srch_ci - data.date_time
    data['time_to_ci'] = data['time_to_ci'].apply(lambda td: td / np.timedelta64(1, 'D'))
    data['time_to_ci'] = data['time_to_ci'].map(bin_time)
    data['time_dest'] = 'time_dest' + data.time_to_ci.map(str) + '*' + data.srch_destination_id.map(str)
    data['time_dest'] = data['time_dest'].map(hash)

    for col in cat_col:
        data[col] = col + data[col].map(str)
        data[col] = data[col].map(hash)


submission = pd.read_csv('input/sample_submission.csv')

cat_col_all = cat_col + ['season_dest', 'time_dest']

#-----------------定义评判标准-----------------
def map5eval(preds, actual):
    predicted = preds.argsort(axis=1)[:,-np.arange(5)]
    metric = 0.
    for i in range(5):
        metric += np.sum(actual==predicted[:,i])/(i+1)
    metric /= actual.shape[0]
    return metric


if os.path.exists('output/probs/sgd.pkl'):
    with open('output/probs/sgd.pkl', 'rb') as f:
        clf = pickle.load(f)
else:
    clf = SGDClassifier(loss='log', n_jobs=-1, alpha=0.0000025, verbose=0)
# clf.sparsify()
for epoch in range(5):
    count = 0
    chunksize = 200000
    n_features = 3000000
    print('Epoch %d started' % epoch)
    reader = pd.read_csv('input/train.csv', parse_dates=['date_time', 'srch_ci', 'srch_co'], chunksize=chunksize)
    for chunk in reader:
        try:
            pre_process(chunk)
            y = chunk.hotel_cluster
            sw = 1 + 4 * chunk.is_booking
            chunk.drop(['cnt', 'hotel_cluster', 'is_booking'], axis=1, inplace=True)

            XN = csr_matrix(chunk[num_col].values)
            X = csr_matrix((chunk.shape[0], n_features))
            rows = np.arange(chunk.shape[0])
            for col in cat_col_all:
                dat = np.ones(chunk.shape[0])
                cols = chunk[col] % n_features
                X += csr_matrix((dat, (rows, cols)), shape=(chunk.shape[0], n_features))
            X = hstack((XN, X))
            book_indices = sw[sw > 1].index.tolist()
            X_test = csr_matrix(X)[book_indices]
            y_test = y[book_indices]

            clf.partial_fit(X, y, classes=np.arange(100), sample_weight=sw)

            count = count + chunksize
            map5 = map5eval(clf.predict_proba(X_test), y_test)
            print('%d rows completed. MAP@5: %f' % (count, map5))
            if (count / chunksize == 200):
                break
        except Exception as e:
            print('Error: %s' % str(e))
            pass

with open('output/probs/sgd.pkl', 'wb') as f:
    pickle.dump(clf, f)

count = 0
chunksize = 10000
n_features = 3000000
preds = np.empty((0, 100))
reader = pd.read_csv('../input/test.csv', parse_dates=['date_time', 'srch_ci', 'srch_co'], chunksize=chunksize)
for chunk in reader:
    chunk.drop(['id'], axis=1, inplace=True)
    pre_process(chunk)

    XN = csr_matrix(chunk[num_col].values)
    X = csr_matrix((chunk.shape[0], n_features))
    rows = np.arange(chunk.shape[0])
    for col in cat_col_all:
        dat = np.ones(chunk.shape[0])
        cols = chunk[col] % n_features
        X += csr_matrix((dat, (rows, cols)), shape=(chunk.shape[0], n_features))
    X = hstack((XN, X))

    pred = clf.predict_proba(X)
    preds = np.vstack((preds, pred))
    count = count + chunksize
    print('%d rows completed' % count)

del clf

if os.path.exists('output/probs/allpreds_sgd.h5'):
    with h5py.File('output/probs/allpreds_sgd.h5', 'r+') as hf:
        predshf = hf['preds']
        print('writing latest probabilities to file')
        predshf[...] = preds
else:
    with h5py.File('output/probs/allpreds_sgd.h5', 'w') as hf:
        print('writing latest probabilities to file')
        hf.create_dataset('preds', data=preds)

col_ind = np.argsort(-preds, axis=1)[:, :5]
hc = [' '.join(row.astype(str)) for row in col_ind]

sub = pd.DataFrame(data=hc, index=submission.id)
sub.reset_index(inplace=True)
sub.columns = submission.columns
sub.to_csv('output/pred_sub.csv', index=False)