import os
import pandas as pd
import random
import tensorflow as tf
import xgboost as xgb
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

from GAN import GAN
from plotc import *


class TrainXGBBoost:

    def __init__(self, num_historical_days, days=10, pct_change=0):
        self.data = []
        self.labels = []
        self.test_data = []
        self.test_labels = []
    #    assert os.path.exists('./models/checkpoint')
        gan = GAN(num_features=5, num_historical_days=num_historical_days,
                        generator_input_size=200, is_train=False)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if os.path.exists(f'xgbbmodels/checkpoint'):
                with open(f'xgbbmodels/checkpoint', 'rb') as f:
                    model_name = next(f).split('"'.encode())[1]
                    saver.restore(sess, "{}xgbbmodels/{}".format('', model_name.decode()))

            files = ['CANBK__EQ__NSE__NSE__MINUTE.csv']      
            for file in files:
                print(file)
                df = pd.read_csv(file, index_col='timestamp', parse_dates=True)
                df = df[['open','high','low','close','volume']]
                labels = df.close.pct_change(days).map(lambda x: int(x > pct_change/100.0))
                df = ((df -
                df.rolling(num_historical_days).mean().shift(-num_historical_days))
                /(df.rolling(num_historical_days).max().shift(-num_historical_days)
                -df.rolling(num_historical_days).min().shift(-num_historical_days)))
                df['labels'] = labels
                df = df.dropna()
                test_df = df[:365]
                df = df[400:]
                data = df[['open', 'high', 'low', 'close', 'volume']].values
                labels = df['labels'].values
                for i in range(num_historical_days, len(df), num_historical_days):
                    features = sess.run(gan.features, feed_dict={gan.X:[data[i-num_historical_days:i]]})
                    self.data.append(features[0])
                    #print(features[0])
                    self.labels.append(labels[i-1])
                data = test_df[['open', 'high', 'low', 'close', 'volume']].values
                labels = test_df['labels'].values
                for i in range(num_historical_days, len(test_df), 1):
                    features = sess.run(gan.features, feed_dict={gan.X:[data[i-num_historical_days:i]]})
                    self.test_data.append(features[0])
                    self.test_labels.append(labels[i-1])



    def train(self):
        if not os.path.exists(f'xgbbmodels'):
            os.makedirs(f'xgbbmodels')
        
        params = {}
        params['objective'] = 'multi:softprob'
        params['eta'] = 0.01
        params['num_class'] = 2
        params['max_depth'] = 20
        params['subsample'] = 0.05
        params['colsample_bytree'] = 0.05
        params['eval_metric'] = 'mlogloss'
        train = xgb.DMatrix(pd.DataFrame(self.data), self.labels)
        test = xgb.DMatrix(pd.DataFrame(self.test_data), self.test_labels)
        #print(train)
        watchlist = [(train, 'train'), (test, 'test')]
        clf = xgb.train(params, train, 10000, evals=watchlist, early_stopping_rounds=100)
        joblib.dump(clf, f'xgbbmodels/clf.pkl')
        cm = confusion_matrix(self.test_labels, list(map(lambda x: int(x[1] > .5), clf.predict(test))))
        print(cm)
        plot_confusion_matrix(cm, ['Down', 'Up'], normalize=True, title="Confusion Matrix")
if __name__ == "__main__":

    tf.reset_default_graph()

    boost_model = TrainXGBBoost(num_historical_days=20, days=10, pct_change=0.1)
    boost_model.train()