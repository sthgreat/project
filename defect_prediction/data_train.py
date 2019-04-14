import pandas as pd
import numpy as np
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from lightgbm.sklearn import LGBMRegressor
import sklearn


def read_data(path):
    try:
        data_train_hash = pd.read_csv(path, encoding='utf-8')
    except:
        data_train_hash = pd.read_csv(path, encoding='gbk')
    data_train_hash = data_train_hash[data_train_hash['label'] > 0]
    train_data_label = data_train_hash['label']
    train_data_features = data_train_hash.drop('label', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(train_data_features, train_data_label, test_size=0.2,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test


def xgb_train(X_train, X_test, y_train, y_test):
    params = {
        'booster': 'gbtree',
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'gamma': 0.1,  # 在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。 这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关，所以是需要调整的。
        # 'max_depth': 5,  # 和GBM中的参数相同，这个值为树的最大深度。 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本。 需要使用CV函数来进行调优。 典型值：3-10
        # 'lambda': 3,
        # 'max_delta_step': 20,  # 这参数限制每棵树权重改变的最大步长。如果这个参数的值为0，那就意味着没有约束。如果它被赋予了某个正值，那么它会让这个算法更加保守。 通常，这个参数不需要设置。但是当各类别的样本十分不平衡时，它对逻辑回归是很有帮助的。 这个参数一般用不到，但是你可以挖掘出来它更多的用处。
        'subsample': 0.8,  # 和GBM中的subsample参数一模一样。这个参数控制对于每棵树，随机采样的比例。 减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。 典型值：0.5-1
        # 'colsample_bytree': 0.7,  # 和GBM里面的max_features参数类似。用来控制每棵随机采样的列数的占比(每一列是一个特征)。 典型值：0.5-1
        # 'min_child_weight': 3,  # 决定最小叶子节点样本权重和。 和GBM的 min_child_leaf 参数类似，但不完全一样。XGBoost的这个参数是最小样本权重的和，而GBM参数是最小样本总数。 这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。 但是如果这个值过高，会导致欠拟合。这个参数需要使用CV来调整。
        # 'silent': 0,
        'eta': 0.05,  # 每一步的权重
        # 'seed': 1000,
        # 'nthread': 12,
    }

    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)
    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    model = xgb.train(params, dtrain, num_boost_round=250000, evals=watchlist, early_stopping_rounds=600)
    model.save_model("./model/xgb.model")
    # model.dump_model('./model/xgb_dump_2.txt', './model/xgb_featmap_2.txt')
    return model


def predict(model_path, test_set_path):
    try:
        data_train_hash = pd.read_csv(test_set_path, encoding='utf-8')
    except:
        data_train_hash = pd.read_csv(test_set_path, encoding='gbk')
    data_train_hash = data_train_hash[data_train_hash['label'] > 0]
    train_data_label = data_train_hash['label']
    train_data_features = data_train_hash.drop('label', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(train_data_features, train_data_label, test_size=0.9,
                                                        random_state=0)
    model = xgb.Booster({'nthread': 4})  # init model
    model.load_model(model_path)  # load data
    dtest = xgb.DMatrix(X_test)
    ans = model.predict(dtest)
    y_test = list(y_test)
    y_test = np.array(y_test)
    error = 0
    for i in range(len(ans)):
        error += ans[i] - y_test[i]
    error = error/len(ans)
    print('平均误差为'+str(error)+'天')


if __name__ == "__main__":
    data_path = './data/data/first_experiment.csv'
    X_train, X_test, y_train, y_test = read_data(data_path)
    xgb_train(X_train, X_test, y_train, y_test)

    # test_set_path = './data/test_time/train_hash_easy_test.csv'
    # model_path = './model/xgb.model'
    # predict(model_path, test_set_path)

    # model = xgb.Booster({'nthread': 4})  # init model
    # model.load_model('./model/xgb.model')  # load data
    # plot_importance(model)
    # plt.show()
