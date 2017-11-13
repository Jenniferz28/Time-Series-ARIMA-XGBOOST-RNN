import pandas as pd
import numpy as np
from matplotlib import dates
import matplotlib.pyplot as plt


def preprocess(N_rows, parse_dates, filename):
    total_rows = sum(1 for l in open(filename))
    variable_names = pd.read_csv(
        filename, header=0, delimiter=';', sep='', nrows=5)
    df = pd.read_csv(filename, header=0, delimiter=';', sep='', names=variable_names.columns,
                     parse_dates=parse_dates, index_col=0, nrows=N_rows, skiprows=total_rows - N_rows)
    df_no_na = df.replace('?', np.NaN)
    df_no_na.dropna(inplace=True)
    return df_no_na.astype(float)


def timeseries_plot(y, color, y_label):
    # y is Series with index of datetime
    days = dates.DayLocator()
    dfmt_minor = dates.DateFormatter('%m-%d')
    weekday = dates.WeekdayLocator(byweekday=(), interval=1)

    fig, ax = plt.subplots()
    ax.xaxis.set_minor_locator(days)
    ax.xaxis.set_minor_formatter(dfmt_minor)

    ax.xaxis.set_major_locator(weekday)
    ax.xaxis.set_major_formatter(dates.DateFormatter('\n\n%a'))

    ax.set_ylabel(y_label)
    ax.plot(y.index, y, color)
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    plt.savefig(y_label + '.png', dpi=300)
    plt.show()

# average time series


def bucket_avg(ts, bucket):
    # ts is Sereis with index
    # bucket =["30T","60T","M".....]
    y = ts.resample(bucket).mean()
    return y


def config_plot():
    plt.style.use('seaborn-paper')
#    plt.rcParams.update({'axes.prop_cycle': cycler(color='jet')})
    plt.rcParams.update({'axes.titlesize': 20})
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams.update({'axes.labelsize': 22})
    plt.rcParams.update({'xtick.labelsize': 16})
    plt.rcParams.update({'ytick.labelsize': 16})
    plt.rcParams.update({'figure.figsize': (10, 6)})
    plt.rcParams.update({'legend.fontsize': 20})
    return 1


# static xgboost
# get one-hot encoder for features
def date_transform(df, encode_cols):
    # extract a few features from datetime
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['WeekofYear'] = df.index.weekofyear
    df['DayofWeek'] = df.index.weekday
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    # one hot encoder for categorical variables
    for col in encode_cols:
        df[col] = df[col].astype('category')
    df = pd.get_dummies(df, columns=encode_cols)
    return df


def get_unseen_data(unseen_start, steps, encode_cols, bucket_size):
    index = pd.date_range(unseen_start,
                          periods=steps, freq=bucket_size)
    df = pd.DataFrame(pd.Series(np.zeros(steps), index=index),
                      columns=['Global_active_power'])
    return df

# dynamic xgboost
# shift 2 steps for every lag


def data_add_timesteps(data, column, lag):
    column = data[column]
    step_columns = [column.shift(i) for i in range(2, lag + 1, 2)]
    df_steps = pd.concat(step_columns, axis=1)
    # current Global_active_power is at first columns
    df = pd.concat([data, df_steps], axis=1)
    return df
