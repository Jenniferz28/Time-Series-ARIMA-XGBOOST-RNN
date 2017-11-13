# basic + dates
import numpy as np
import pandas as pd
from util import *
import matplotlib.pyplot as plt
import seaborn as sn




parse_dates = [['Date', 'Time']]
filename = "household_power_consumption.txt"
encode_cols = ['Month', 'DayofWeek', 'Hour']
bucket_size="60T"

# (1)  data is not datetime ordered. It is random
N_rows = 60000
df = preprocess(N_rows, parse_dates, filename)

G_power=df["Global_active_power"]
#G_power_sort=G_power.sort_values('index')

df = pd.DataFrame(bucket_avg(G_power,bucket_size))
df.dropna(inplace=True)
df.Global_active_power.plot(style='b.')
plt.ylabel('Global Active Power')
plt.xlabel('Time')
plt.title("Last 60000 rows")
plt.savefig( 'EDA_2010.png', dpi=300)
plt.show()


### (2)So we focus on the 2010-11 area which is around the last 19000 data rows
N_rows = 19000
df = preprocess(N_rows, parse_dates, filename)
G_power=df["Global_active_power"]
print(G_power.shape) # 21661 rows
df_G = pd.DataFrame(bucket_avg(G_power,bucket_size))
df_G.dropna(inplace=True)

# time series plot
ts_label='Global_active_power_in_Nov_2010'
timeseries_plot(df_G.Global_active_power,'g', ts_label)

# component plot
from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 11, 9
decomposition = sm.tsa.seasonal_decompose(df_G.Global_active_power, model='additive')
fig = decomposition.plot()
plt.tight_layout()
plt.savefig( 'ts_decomposition_plot.png', dpi=300)
plt.show()


# heatmap
df_G['Time of Day'] = df_G.index.time
df_G['Date'] = df_G.index.date
print(df_G.head())
dfG_pivot = df_G.pivot_table(index="Date", columns="Time of Day",values='Global_active_power',fill_value=0)
dfG_pivot.head()
plt.figure(figsize=(12, 8))
sn.heatmap(dfG_pivot,cmap='Blues')
plt.tight_layout()
plt.savefig( 'date_time_power_heatmap.png', dpi=300)
plt.show()

# correlation plot
names = df.columns
correlations = df.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1,cmap='Accent')
fig.colorbar(cax)
ticks = np.arange(0,7,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names, rotation = 90)
ax.set_yticklabels(names)
#plt.figure(figsize=(8,8))
plt.tight_layout()
plt.savefig('correlation.png', dpi=300)
plt.show()
