import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

dir = 'C:/Users/15/Desktop/DataSet/'
# CSV 파일 읽기
data = pd.read_csv(dir+"/[Dataset] Module 18 (weatherHistory).csv")

# 처음 10개 값 출력
print(data.head(10))

print(data.dtypes)

# 0 및 NaN 값을 제거하는 데이터 처리
data= data.replace(0,np.nan)

pressure = data['Pressure (millibars)']
humidity = data['Humidity']
temperature = data['Temperature (C)']
#wind_speed = data['Wind Speed (km/h)']
#visibility = data['Visibility (km)']

plt.figure(figsize=(10,10))
ax = sns.scatterplot(x=temperature, y=humidity)
plt.show()

plt.figure(figsize=(10,10))
ax = sns.scatterplot(x=humidity, y=pressure)
plt.show()

plt.figure(figsize=(10,10))
ax = sns.scatterplot(x=temperature, y=pressure)
plt.show()

snsplot = sns.heatmap(data.corr(), annot=True,linewidths=0.5, fmt='.1f')
#snsplot.figure.savefig("corr_data.png")

plt.show()

data['Formatted Date'] =  pd.to_datetime(data['Formatted Date'],
                              format='%Y-%m-%d %H:%M:%S.%f', utc=True)
data = data.drop("Loud Cover", axis = 1)

data = data.dropna()
# 날짜 열이 날짜, 시간 형식인지 확인
print(data.info())
print(data['Formatted Date'][10].month)
print(data['Formatted Date'][0].day)
print(data['Formatted Date'][0].month)
print(data['Formatted Date'][0].year)
data['month'] = data['Formatted Date'].dt.month


data_jan = data[data.month == 1]
data_feb = data[data.month == 2]
data_mar = data[data.month == 3]
data_apr = data[data.month == 4]
data_may = data[data.month == 5]
data_jun = data[data.month == 6]
data_jul = data[data.month == 7]
data_aug = data[data.month == 8]
data_sep = data[data.month == 9]
data_oct = data[data.month == 10]
data_nov = data[data.month == 11]
data_dec = data[data.month == 12]

snsplot = sns.heatmap(data_jan.corr(), annot=True,linewidths=0.5, fmt='.1f')
#snsplot.figure.savefig("corr_data.png")

plt.show()

snsplot = sns.heatmap(data_feb.corr(), annot=True,linewidths=0.5, fmt='.1f')
#snsplot.figure.savefig("corr_data.png")

plt.show()

snsplot = sns.heatmap(data_mar.corr(), annot=True,linewidths=0.5, fmt='.1f')
#snsplot.figure.savefig("corr_data.png")

plt.show()

pressure_sep = data_sep['Pressure (millibars)']
humidity_sep = data_sep['Humidity']
temperature_sep = data_sep['Temperature (C)']

plt.figure(figsize=(10,10))
ax = sns.scatterplot(x=temperature_sep, y=humidity_sep)
plt.show()

features_available = [
    'Temperature (C)',
]
X = data[features_available]
y = data['Humidity']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
linreg = LinearRegression()
linreg.fit(X_train,y_train)

y_pred = linreg.predict(X_test)
df = pd.DataFrame(columns=['Actual', 'Predicted'])
df['Actual'] = y_test
df['Predicted'] = y_pred
print(df.head())

plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

print ("Linear Regression Score")
predict = linreg.predict(X_test)
print ("Mean Absolute Error: ", mean_absolute_error(y_test,y_pred))
print ("Mean Squared Error: ", mean_squared_error(y_test,y_pred))
print ("R2: ", r2_score(y_test,y_pred))