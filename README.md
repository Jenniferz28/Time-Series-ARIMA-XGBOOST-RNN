# Time Series Prediction for individual household power
Individual household power prediction
Dateset: https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

The electric power consumption in an individual household with a one-minute sampling rate over a period between Dec 2006
and Nov 2010 (47 months) were measured. Different electrical quantities and some sub-metering values are recorded. 
Our goal is to predict the Global active power at least 1 hour into the future.

<> There are some data points of Dec 2010, which is not consistent with the data description.

Six independentvariables and a numerical dependent variable Global active power with 2,075,259 observations are available. 
Approximately 1.25% missing values are represented as “?” in the raw dataset. There are several methods to impute the missing
values as the column mean, median or using KNN based approaches. However, missing values are dropped for sim- plicity.
Furthermore, we find that not all observations are ordered by the date time. Therefore we analyze the data with explicit time 
stamp as an index. In the preprocessing step, we perform a bucket-average of the raw data to reduce the noise from the one-
minute sampling rate. For simplicity, we only focus on the last 18000 rows of raw dataset (the most recent data in Nov 2010).

### Here, I used regression tree-based xgboost and univariate time series ARIMA to model the pattern of power consumption.

### Future work should focus on the following two aspects. 

#### (i) Dynamic regression time series model
Given the strong correlations between Sub metering 1, Sub metering 2 and Sub metering 3 and our target variable, 
these variables could be included into the dynamic regression model or regression time series model.

#### (ii) Dynamic xgboost model
Include the timestep-shifted Global active power columns as features. The target variable will be current Global active power. 
Recent history of Global active power up to this time stamp (say, from 100 timesteps before) should be included
as extra features.
