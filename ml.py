import os, csv

#with open("./data/sample_version2.xlsx") as f:
#	## Read parameters here

Tweets = {"4": {"class":{"AutoDriving": {"Apple": [20, 50, 80], "Tesla": [30, 60, 90], "Google": [40, 70, 100]}, "Ecar": {"AppleE": [20, 50, 80], "TeslaE": [30, 60, 90], "GoogleE": [40, 70, 100]}}, "Sentiment": 0.978}}

## Learn to predict the future stock price in training
prev_stocks = []
fut_stocks = []

for k, v in Tweets.items():
	company = v["class"] ##class: **
	for kc, vc in company.items():
		for ks, vs in vc.items():
			price = vs[:-1]
			price.append(v["Sentiment"])
			prev_stocks.append(price)
			fut_stocks.append(vs[-1])

print(prev_stocks) ## Input
print(fut_stocks)  ## Label

## Split Train & Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(prev_stocks, fut_stocks, random_state=1)

print("X_train", X_train)
print("X_test", X_test)
print("y_train", y_train)
print("y_test", y_test)

## Start Regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)

print("Statistic for the regression model:")
print("Score[the coefficient of determination of the prediction]:", reg.score(X_train, y_train))

## Test if the stock price will go up/down in inference
prediction = reg.predict(X_test)
print("prediction:", prediction)

from sklearn.metrics import confusion_matrix
print("Confusion Matrix:", confusion_matrix(y_test, prediction))