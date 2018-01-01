# TensorFlow1.4
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from tensorflow.contrib.timeseries.python.timeseries import estimators as ts_estimators
from tensorflow.contrib.timeseries.python.timeseries import model as ts_model
from tensorflow.contrib.timeseries.python.timeseries import  NumpyReader

from lstm import _LSTMModel
from util import preprocess, bucket_avg
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

# get LSTM data
def get_rnn_data(N_rows, bucket_size):
    parse_dates = [['Date', 'Time']]
    filename = "household_power_consumption.txt"
    df = preprocess(N_rows, parse_dates, filename)
    df = pd.DataFrame(bucket_avg(df["Global_active_power"], bucket_size))
    df.dropna(inplace=True)
    x = np.array(range(df.shape[0]))
    y = np.array(df.Global_active_power)
    return x, y

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  x, y = get_rnn_data(18000, "15T")
  x_train, y_train = x[:900],y[:900] # first 900 data points for training
  x_eval, y_eval = x[900:],y[900:] # last 300 data points for evaluation

  data_train = {
      tf.contrib.timeseries.TrainEvalFeatures.TIMES: x_train,
      tf.contrib.timeseries.TrainEvalFeatures.VALUES: y_train,
  }
  
  data_eval = {
      tf.contrib.timeseries.TrainEvalFeatures.TIMES: x_eval,
      tf.contrib.timeseries.TrainEvalFeatures.VALUES: y_eval,
  }  

  reader = NumpyReader(data_train)
  reader_eval = NumpyReader(data_eval)

  train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
      reader, batch_size=4, window_size=100)

  estimator = ts_estimators.TimeSeriesRegressor(
      model=_LSTMModel(num_features=1, num_units=128),
      optimizer=tf.train.AdamOptimizer(0.001))

  estimator.train(input_fn=train_input_fn, steps=2000)
  evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader_eval)
  evaluation = estimator.evaluate(input_fn=evaluation_input_fn, steps=1)
  # Predict starting after the evaluation
  (predictions,) = tuple(estimator.predict(
      input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
          evaluation, steps=200)))

  observed_times = x
  observed = y
  evaluated_times = evaluation["times"][0]
  evaluated = evaluation["mean"][0]
  predicted_times = predictions['times']
  predicted = predictions["mean"]

  plt.figure(figsize=(15, 8))
  plt.axvline(1200, linestyle="dotted", linewidth=4, color='r')
  observed_lines = plt.plot(observed_times, observed, label="observation", color="k")
  evaluated_lines = plt.plot(evaluated_times, evaluated, label="evaluation", color="g")
  predicted_lines = plt.plot(predicted_times, predicted, label="forecasts", color="r")
  xlim = ((0,1401))
  plt.legend(handles=[observed_lines[0], evaluated_lines[0], predicted_lines[0]],
             loc="upper left")
  plt.tight_layout()
  plt.savefig('predict_result.png',dpi=300)
