import tensorflow as tf
tfgan = tf.contrib.gan
from GAN import TrainGan,GAN
from CNN import TrainCNN,CNN
from plotc import *
from XGBB import TrainXGBBoost
gan_estimator = tfgan.estimator.GANEstimator(
         model_dir = '/checkpoint',
         generator_fn=GAN(num_features=5, num_historical_days=20,generator_input_size=200),
         discriminator_fn=TrainCNN(num_historical_days=20, days=5, pct_change=0.1),
         generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
         discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
         generator_optimizer=tf.compat.v1.train.AdamOptimizer(0.1, 0.5),
         discriminator_optimizer=tf.compat.v1.train.AdamOptimizer(0.1, 0.5))

import os
import pandas as pd
import random
import tensorflow as tf
import xgboost as xgb
from sklearn.externals import joblib

class Predict:
  def __init__(self, num_historical_days=20, days=10, pct_change=0, gan_model=f'models/', cnn_modle=f'cnn_models/', xgb_model=f'xgbbmodels/clf.pkl'):
    self.data = []
    self.num_historical_days = num_historical_days
    self.gan_model = gan_model
    self.cnn_modle = cnn_modle
    self.xgb_model = xgb_model
    
    files = ['CANBK__EQ__NSE__NSE__MINUTE.csv'] 
    for file in files:
      
      print(file)
      df = pd.read_csv(file, index_col='timestamp', parse_dates=True)
      df = df[['open','high','low','close','volume']]
            
      df = ((df -
            df.rolling(num_historical_days).mean().shift(-num_historical_days))
            /(df.rolling(num_historical_days).max().shift(-num_historical_days)
            -df.rolling(num_historical_days).min().shift(-num_historical_days)))
      df = df.dropna()
      self.data.append((file.split('/')[-1], df.iloc[0], df[200:200+num_historical_days].values))
      
      
  def gan_predict(self):
    tf.reset_default_graph()
    gan = GAN(num_features=5, num_historical_days=self.num_historical_days, generator_input_size=200, is_train=False)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver()
      filename = ""
      with open(f'models/checkpoint', 'rb') as f:
                model_name = next(f).split('"'.encode())[1]
                filename = "{}models/{}".format('', model_name.decode())
      saver.restore(sess, "{}".format(filename))
      clf = joblib.load(self.xgb_model)
      for sym, date, data in self.data:
        features = sess.run(gan.features, feed_dict={gan.X:[data]})
        print(len(features[0]))
        
        features = xgb.DMatrix(pd.DataFrame(features))
        print('{} {} {}'.format(str(date).split(' ')[0], sym, clf.predict(features)[0][1] > 0.5))
        print(clf.predict(features))

p = Predict()
p.gan_predict()
