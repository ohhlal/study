import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox     # 白噪声检验
import matplotlib.pyplot as plt

data = [10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422,
6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331]
data = pd.Series(data)
# print(data)
data.index = pd.Index([1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
                       2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
# print(data)
# data.plot()


# plot_acf(data)


diff_1 = data.diff(1).dropna()
# print(diff_1)
# diff_1.plot()


# 白噪声 = acorr_ljungbox(data)
# 白噪声_差分 = acorr_ljungbox(diff_1)
# print(白噪声)
# print(白噪声_差分)

# plt.figure('diff_1_acf')
# plot_acf(diff_1)

# plt.figure('diff_1_pacf')
plot_pacf(diff_1, lags=10)
model = ARIMA(data, order=(0, 1, 1))
model_fit = model.fit()
print(model_fit.summary())
# plt.show()





