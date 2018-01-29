from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from util import preprocess, bucket_avg

import tensorflow as tf

from tensorflow.contrib.timeseries.python.timeseries import  NumpyReader

from tensorflow.contrib.timeseries.python.timeseries import estimators as ts_estimators
from tensorflow.contrib.timeseries.python.timeseries import model as ts_model

from lstm import _LSTMModel
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt



# get data

def get_rnn_data(N_rows, bucket_size):
    parse_dates = [['Date', 'Time']]
    filename = "household_power_consumption.txt"
    df = preprocess(N_rows, parse_dates, filename)
    global_power=pd.DataFrame(bucket_avg(df["Global_active_power"], bucket_size))
    sub1=pd.DataFrame(bucket_avg(df["Sub_metering_1"], bucket_size))
    sub2=pd.DataFrame(bucket_avg(df["Sub_metering_2"], bucket_size))
    sub3=pd.DataFrame(bucket_avg(df["Sub_metering_3"], bucket_size))

    #df.dropna(inplace=True)
    #df.iloc[-1, :].index  # last time step  #2010-11-26 21:00:00
    x = np.array(range(global_power.shape[0]))
    y = np.column_stack((sub1, sub2, sub3, global_power))
    return x, y


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  x, y = get_rnn_data(18000, "15T")
  data = {
      tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
      tf.contrib.timeseries.TrainEvalFeatures.VALUES: y,
  }
  reader = NumpyReader(data)
  train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
      reader, batch_size=4, window_size=100)

  estimator = ts_estimators.TimeSeriesRegressor(
      model=_LSTMModel(num_features=4, num_units=128),
      optimizer=tf.train.AdamOptimizer(0.001))

  estimator.train(input_fn=train_input_fn, steps=1000)
  evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
  evaluation = estimator.evaluate(input_fn=evaluation_input_fn, steps=1)
  # Predict starting after the evaluation
  (predictions,) = tuple(estimator.predict(
      input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
          evaluation, steps=100)))

  observed_times = evaluation["times"][0]
  observed = evaluation["observed"][0, :, :]
  evaluated_times = evaluation["times"][0]
  evaluated = evaluation["mean"][0]
  predicted_times = predictions['times']
  predicted = predictions["mean"]
#plot all 4 variables
  plt.figure(figsize=(15, 16))
  plt.subplot(411)
  plt.axvline(1200, linestyle="dotted", linewidth=4, color='r')
  observed_lines = plt.plot(observed_times, observed[:,3], label="observation", color="k")
  evaluated_lines = plt.plot(evaluated_times, evaluated[:,3], label="evaluation", color="g")
  predicted_lines = plt.plot(predicted_times, predicted[:,3], label="prediction", color="r")
  plt.legend(handles=[observed_lines[0], evaluated_lines[0], predicted_lines[0]],loc="upper left")
  plt.title("Global active power")
  
  plt.subplot(412)
  plt.axvline(1200, linestyle="dotted", linewidth=4, color='r')
  observed_lines = plt.plot(observed_times, observed[:,0], label="observation", color="k")
  evaluated_lines = plt.plot(evaluated_times, evaluated[:,0], label="evaluation", color="g")
  predicted_lines = plt.plot(predicted_times, predicted[:,0], label="prediction", color="r")
  plt.legend(handles=[observed_lines[0], evaluated_lines[0], predicted_lines[0]],loc="upper left")
  plt.title("Sub_metering 1")
  
  plt.subplot(413)
  plt.axvline(1200, linestyle="dotted", linewidth=4, color='r')
  observed_lines = plt.plot(observed_times, observed[:,1], label="observation", color="k")
  evaluated_lines = plt.plot(evaluated_times, evaluated[:,1], label="evaluation", color="g")
  predicted_lines = plt.plot(predicted_times, predicted[:,1], label="prediction", color="r")
  plt.legend(handles=[observed_lines[0], evaluated_lines[0], predicted_lines[0]],loc="upper left")
  plt.title("Sub_metering 2")  
  
  plt.subplot(414)
  plt.axvline(1200, linestyle="dotted", linewidth=4, color='r')
  observed_lines = plt.plot(observed_times, observed[:,2], label="observation", color="k")
  evaluated_lines = plt.plot(evaluated_times, evaluated[:,2], label="evaluation", color="g")
  predicted_lines = plt.plot(predicted_times, predicted[:,2], label="prediction", color="r")
  plt.legend(handles=[observed_lines[0], evaluated_lines[0], predicted_lines[0]],loc="upper left")
  plt.title("Sub_metering 3")  
  
  plt.savefig('multivariate.png',dpi=300)