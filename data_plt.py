# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge,LinearRegression,Ridge,Lasso
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import warnings
import re
import plotly.offline as py
py.init_notebook_mode(connected=True)
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns',None)
pd.set_option('max_colwidth',100)

train = pd.read_csv('./jinnan_round1_train_20181227.csv')
test = pd.read_csv('./jinnan_round1_testB_20190121.csv')


class grid():
    def __init__(self, model):
        self.model = model

    def grid_get(self, X, y, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X, y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])

def get_phase(t1,t2):
    try:
        h1, m1, s1=t1.split(':')
        h2, m2, s2=t2.split(':')
    except:
        if t1 == -1 or t2 == -1:
            return -1
    if int(h2) >= int(h1):
        tm = (int(h2) * 3600 + int(m2) * 60 - int(m1) * 60 - int(h1) * 3600) / 3600
    else:
        tm = (int(h2) * 3600 + int(m2) * 60 - int(m1) * 60 - int(h1) * 3600) / 3600 + 24
    return tm


def timeTranSecond(t):
    try:
        t, m, s = t.split(":")
    except:
        if t == '1900/1/9 7:00':
            return 7 * 3600 / 3600
        elif t == '1900/1/1 2:30':
            return (2 * 3600 + 30 * 60) / 3600
        elif t == -1:
            return -1
        else:
            return 0
    try:
        tm = (int(t) * 3600 + int(m) * 60 + int(s)) / 3600
    except:
        return (30 * 60) / 3600
    return tm


def getDuration(se):
    try:
        sh, sm, eh, em = re.findall(r"\d+\.?\d*", se)
    except:
        if se == -1:
            return -1
    try:
        if int(sh) > int(eh):
            tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600 + 24
        else:
            tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600
    except:
        if se == '19:-20:05':
            return 1
        elif se == '15:00-1600':
            return 1
    return tm


def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse


train.loc[train['B14'] == 40, 'B14'] = 400
train.drop(train[train['收率'] < 0.87].index, inplace=True)
full = pd.concat([train, test], ignore_index=True)
cols = ["A2", "A3", "A4"]
for col in cols:
    full[col].fillna(0, inplace=True)
cols1 = ["A7", "A8", "B10", "B11", "A20", "A24", "A26"]
for col in cols1:
    full[col].fillna(-1, inplace=True)
cols2 = ["B1", "B2", "B3", "B8", "B12", "B13", "A21", "A23"]
for col in cols2:
    full[col].fillna(full[col].mode()[0], inplace=True)
full['a21_a22_a23'] = full['A21']+full['A22']+full['A23']
cols3 = ["A25", "A27"]
for col in cols3:
    full[col] = full.groupby(['a21_a22_a23'])[col].transform(lambda x: x.fillna(x.median()))


full['a1_a3_a4']=full['A1']+full['A3']+full['A4']
full['a1_a3']=full['A1']+full['A3']
full['a1_a4']=full['A1']+full['A4']

full['a10_a6']=full['A10']-full['A6']
full['a12_a10']=full['A12']-full['A10']
full['a15_a12']=full['A15']-full['A12']
full['a17_a15']=full['A17']-full['A15']
full['a27_a25']=full['A27']-full['A25']
full['b6_b8']=full['B6']-full['B8']

full['a10_a6/a9_a5']=(full['A10']-full['A6'])/full.apply(lambda df:get_phase(df['A5'],df['A9']),axis=1)
full['a12_a10/a11_a9']=(full['A12']-full['A10'])/full.apply(lambda df:get_phase(df['A9'],df['A11']),axis=1)
full['a15_a12/a14_a11']=(full['A15']-full['A12'])/full.apply(lambda df:get_phase(df['A11'],df['A14']),axis=1)
full['a17_a15/a16_a14']=(full['A17']-full['A15'])/full.apply(lambda df:get_phase(df['A14'],df['A16']),axis=1)
full['a27_a25/a26_a24']=(full['A27']-full['A25'])/full.apply(lambda df:get_phase(df['A24'],df['A26']),axis=1)
full['b6_b8/b7_b5']=(full['B6']-full['B8'])/full.apply(lambda df:get_phase(df['B5'],df['B7']),axis=1)

full['b14/a1_a3_a4_a19_b1_b12'] = full['B14']/(full['A1']+full['A3']+full['A4']+full['A19']+full['B1']+full['B12'])
full['b14/a1_a3_a4_a19_b1_b12_b14'] = full['B12']/(full['A1']+full['A3']+full['A4']+full['A19']+full['B1']+full['B14'])

for f in ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']:
    try:
        full[f] = full[f].apply(timeTranSecond)
    except:
        print(f, '应该在前面被删除了！')

for f in ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']:
    full[f] = full.apply(lambda df: getDuration(df[f]), axis=1)
full['样本id'] = full['样本id'].apply(lambda x: int(x.split('_')[1]))
good_cols=list(full.columns)
good_cols.remove('样本id')
good_cols.remove('收率')
# for f in good_cols:
#     full[f] = full[f].map(dict(zip(full[f].unique(), range(0, full[f].nunique()))))

n_train=train.shape[0]
X = full[:n_train]
test_X = full[n_train:]
y= X.收率
X.drop(['收率'], axis=1, inplace=True)
test_X.drop(['收率'], axis=1, inplace=True)
# X_train = X[list(X.columns)].values
# X_test = test_X[list(X.columns)].values
# y_train = y.values

# # grid(Lasso()).grid_get(X,y,{'alpha': [0.02,0.0002,0.000222,0.0000224],'max_iter':[10000]})
# grid(xgb.XGBRegressor()).grid_get(X_train,y_train,{'num_leaves': [100],
#         'min_data_in_leaf': [9],
#         'objective': ['regression'],
#         'max_depth': [-1],
#         'learning_rate': [0.01],
#         'min_child_samples': [15],
#         "boosting": ['gbdt'],
#         "feature_fraction": [0.9],
#         "bagging_freq": [1],
#         "bagging_fraction": [0.9],
#         "bagging_seed": [5,13,40,50],
#         "metric": ['mse'],
#         "lambda_l1": [0.000001],
#         'verbosity': [-1]})
# grid(xgb.XGBRegressor()).grid_get(X_train,y_train,{'eta': [0.1], 'max_depth': [6], 'subsample': [0.9],
#                                                    'colsample_bytree': [0.5],'objective': ['reg:linear'],
#                                                    'eval_metric': ['rmse'], 'silent': [True], 'nthread': [3]})
X_train = X[list(X.columns)].values
X_test = test_X[list(X.columns)].values
# one hot
enc = OneHotEncoder()
# for f in good_cols:
#     enc.fit(full[f].values.reshape(-1, 1))
#     X_train = sparse.hstack((X_train, enc.transform(X[f].values.reshape(-1, 1))), 'csr')
#     X_test = sparse.hstack((X_test, enc.transform(test_X[f].values.reshape(-1, 1))), 'csr')
print(X_train.shape)
print(X_test.shape)

y_train = y.values
#
param = {'num_leaves': 100,
        'min_data_in_leaf': 9,
        'objective': 'regression',
        'max_depth': -1,
        'learning_rate': 0.01,
        "min_child_samples": 15,
        "boosting": "gbdt",
        "feature_fraction": 0.9,
        "bagging_freq": 1,
        "bagging_fraction": 0.9,
        "bagging_seed": 13,
        "metric": 'mse',
        "lambda_l1": 0.000001,
        "verbosity": -1}
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 3000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=200,
                    early_stopping_rounds=100)
    oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)

    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, y)))

##### xgb
xgb_params = {'eta': 0.1, 'max_depth': 6, 'subsample': 0.9, 'colsample_bytree': 0.5,
              'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 3}

folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_xgb = np.zeros(len(train))
predictions_xgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                    verbose_eval=100, params=xgb_params)
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
    predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, y)))

# 将lgb和xgb的结果进行stacking
train_stack = np.vstack([oof_lgb, oof_xgb]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
oof_stack1 = np.zeros(train_stack.shape[0])
predictions1 = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, y)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], y.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], y.iloc[val_idx].values

    clf_3 = BayesianRidge()
    clf_3.fit(trn_data, trn_y)

    oof_stack1[val_idx] = clf_3.predict(val_data)
    predictions1 += clf_3.predict(test_stack) / 10

print("CV score: {:<8.8f}".format(mean_squared_error(y.values, oof_stack1)))

sub_df = pd.DataFrame()
sub_df[0] = pd.read_csv('./jinnan_round1_testB_20190121.csv', header=None)[0][1:]
sub_df[1] = predictions1
sub_df[1] = sub_df[1].apply(lambda x:round(x, 3))
sub_df.to_csv('./prediction.csv', index=False, header=None)

