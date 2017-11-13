import pandas as pd
from util import timeseries_plot, bucket_avg, preprocess, config_plot
from myArima import *

config_plot()


# we focus on the last 10 days data in Nov 2010
N_rows = 15000
parse_dates = [['Date', 'Time']]
filename = "household_power_consumption.txt"

df = preprocess(N_rows, parse_dates, filename)

G_power = pd.to_numeric(df["Global_active_power"])
# time series plot of one-minute sampling rate data
timeseries_plot(G_power, 'g', 'Global_active_power')

# we take a 30 minutes bucket average of our time series data to reduce noise.
bucket_size = "30T"
G_power_avg = bucket_avg(G_power, bucket_size)
# plot of 30 minutes average.
ts_label = 'G_power_avg'
timeseries_plot(G_power_avg, 'g', ts_label)


# "Grid search" of seasonal ARIMA model.
# the seasonal periodicy 24 hours, i.e. S=24*60/30 = 48 samples
arima_para = {}
arima_para['p'] = range(2)
arima_para['d'] = range(2)
arima_para['q'] = range(2)
# the seasonal periodicy is  24 hours
seasonal_para = round(24 * 60 / (float(bucket_size[:-1])))
arima = Arima_Class(arima_para, seasonal_para)

arima.fit(G_power_avg)

# Prediction on observed data starting on pred_start
# observed and prediction starting dates in plots
plot_start = '2010-11-24 00:00:00'
pred_start = '2010-11-25 14:00:00'

# One-step ahead forecasts
dynamic = False
arima.pred(G_power_avg, plot_start, pred_start, dynamic, ts_label)

# Dynamic forecasts
dynamic = True
arima.pred(G_power_avg, plot_start, pred_start, dynamic, ts_label)

# Forecasts to unseen future data
n_steps = 100  # next 100 * 30 min = 50 hours
arima.forcast(G_power_avg, n_steps, ts_label)
