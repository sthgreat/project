import pandas as pd
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from lightgbm.sklearn import LGBMRegressor


def read_data(path):
    data_train_hash = pd.read_csv(path)
    data_train_hash = data_train_hash[data_train_hash['label'] > 0]
    train_data_label = data_train_hash['label']
    train_data_features = data_train_hash.drop('label', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(train_data_features, train_data_label, test_size=0.1,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test


def xgb_train(X_train, X_test, y_train, y_test):
    params = {
        'booster': 'gbtree',
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        # 'gamma': 0.1,
        # 'max_depth': 5,
        'lambda': 3,
        # 'subsample': 0.7,
        # 'colsample_bytree': 0.7,
        # 'min_child_weight': 3,
        # 'silent': 0,
        # 'eta': 0.1,
        # 'seed': 1000,
        'nthread': 12,
    }

    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)
    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    model = xgb.train(params, dtrain, num_boost_round=8000, evals=watchlist, early_stopping_rounds=500)
    model.save_model("./model/xgb_model_3(no name).model")
    # model.dump_model('./model/xgb_dump_2.txt', './model/xgb_featmap_2.txt')
    plot_importance(model)  # 显示特征重要度
    plt.show()
    return model


def predict(model_path, X_test):
    model = xgb.Booster({'nthread': 4})  # init model
    model.load_model(model_path)  # load data


def lightgbm_train(X_train, X_test, y_train, y_test):
    dtrain = lgb.Dataset(X_train, y_train)
    dtest = lgb.Dataset(X_test, y_test, reference=dtrain)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'regression',  # 目标函数
        'metric': 'rmse',  # 评估函数
        'num_leaves': 30,  # 叶子节点数
        'learning_rate': 0.05,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        'nthread': 4
    }
    model = lgb.train(params,dtrain,num_boost_round=50000,valid_sets=dtest,early_stopping_rounds=
                      500)
    model.save_model('./model/lightgbm.model')

    print('Start predicting...')
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)


if __name__ == "__main__":
    data_path = './data/first_experiment_noname.csv'
    X_train, X_test, y_train, y_test = read_data(data_path)
    lightgbm_train(X_train, X_test, y_train, y_test)
    # model = xgb.Booster({'nthread': 4})  # init model
    # model.load_model('./model/xgb.model')  # load data
    # plot_importance(model)
    # plt.show()
