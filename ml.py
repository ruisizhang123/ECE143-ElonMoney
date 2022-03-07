import os, csv
import numpy as np
import torch
import torch.nn as nn

input_dim = 6
hidden_dim = 16
num_layers = 3
output_dim = 3
num_epochs = 100

### Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = x.view(len(x), 1, -1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

### Preprocess csv file into dictionary
import pandas as pd
df = pd.read_csv("./data/newstock111.csv") # read data
df['price-5'] = df['price-5'].str.split(',')
df['price-4'] = df['price-4'].str.split(',')
df['price-3'] = df['price-3'].str.split(',')
df['price-2'] = df['price-2'].str.split(',')
df['price-1'] = df['price-1'].str.split(',')
df['price0'] = df['price0'].str.split(',')
df['price1'] = df['price1'].str.split(',')
df['price2'] = df['price2'].str.split(',')
df['price3'] = df['price3'].str.split(',')
df['Company'] = df['Company'].apply(lambda x: x if (x =='BitCoin' or x == 'DogeCoin' or x == 'Ethereum') else x[2:-2]).str.split('\', \'')

tweets = dict() # initialize dictionary
sentiment_reader = {'Positive' : 1, 'Negative' : -1, 'Neutral' : 0}
for i in range(2500):
    if not df[df.get('ID') == i].empty:
        # add ID and classes to dictionary
        temp = df[df.get('ID')==i]
        # add all data in each category into a list
        classes = temp['Class'].values.tolist()
        companies = temp['Company'].values.tolist()
        price_5 = temp['price-5'].values.tolist()
        price_4 = temp['price-4'].values.tolist()
        price_3 = temp['price-3'].values.tolist()
        price_2 = temp['price-2'].values.tolist()
        price_1 = temp['price-1'].values.tolist()
        price0 = temp['price0'].values.tolist()
        price1 = temp['price1'].values.tolist()
        price2 = temp['price2'].values.tolist()
        price3 = temp['price3'].values.tolist()
        # prepare dictionaries for each class
        tweets[temp.iloc[0].get('ID')] = dict()
        tweets[temp.iloc[0].get('ID')]['class'] = dict()
        for i in range(len(classes)): 
            tweets[temp.iloc[0].get('ID')]['class'][classes[i]] = dict()
            for j in range(len(companies[i])):
                # add stock data to dictionary
                tweets[temp.iloc[0].get('ID')]['class'][classes[i]][companies[i][j]] = [float(price_5[i][j]), float(price_4[i][j]), float(price_3[i][j]), 
                    float(price_2[i][j]), float(price_1[i][j]), float(price0[i][j]), float(price1[i][j]), float(price2[i][j]), float(price3[i][j])]
        # postive number = positive sentiment, negative number = negative sentiment, neutral sentiment = 0
        tweets[temp.iloc[0].get('ID')]['Sentiment'] = sentiment_reader[temp.iloc[0].get('Sentiment')] * temp.iloc[0].get('Confidence') 

        
### Extract data and convet into training & test data
pos_prev_stocks = []
pos_fut_stocks = []

neu_prev_stocks = []
neu_fut_stocks = []

neg_prev_stocks = []
neg_fut_stocks = []

for k, v in tweets.items():
	company = v["class"] ##class: **
	for kc, vc in company.items():
		for ks, vs in vc.items():
			price = vs[:-3]
			if v["Sentiment"] > 0:
				pos_prev_stocks.append(price)
				pos_fut_stocks.append(vs[-3:])
			if v["Sentiment"] == 0:
				neu_prev_stocks.append(price)
				neu_fut_stocks.append(vs[-3:])
			if v["Sentiment"] < 0:
				neg_prev_stocks.append(price)
				neg_fut_stocks.append(vs[-3:])

### linear regression code
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pos_prev_stocks, pos_fut_stocks, random_state=1)
reg = LinearRegression().fit(X_train, y_train)
prediction = reg.predict(X_test)

os.makedirs("./pos_linear", exist_ok = True) 
import matplotlib.pyplot as plt
step1 = [-5,-4,-3,-2,-1,0,1,2,3]
for index in range(len(X_test)): # pos: 2151 | 1184 neg:  | 
    real1 = X_test[index] + y_test[index]
    pred1 = X_test[index] + prediction[index].tolist()
    plt.plot(step1, real1, label='real data')
    plt.plot(step1, pred1, label='prediction data')
    plt.xlabel('time')
    plt.ylabel('price')
    plt.legend()
    plt.savefig('./pos_linear/pred'+str(index)+'.png')
    plt.clf()


### Training neutral twitter model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(neu_prev_stocks, neu_fut_stocks, random_state=1)
reg = LinearRegression().fit(X_train, y_train)
prediction = reg.predict(X_test)
os.makedirs("./neu_linear", exist_ok = True) 

import matplotlib.pyplot as plt
step1 = [-5,-4,-3,-2,-1,0,1,2,3]
for index in range(len(X_test)): # pos: 2151 | 1184 neg:  | 
    real1 = X_test[index] + y_test[index]
    pred1 = X_test[index] + prediction[index].tolist()
    plt.plot(step1, real1, label='real data')
    plt.plot(step1, pred1, label='prediction data')
    plt.xlabel('time')
    plt.ylabel('price')
    plt.legend()
    plt.savefig('./neu_linear/pred'+str(index)+'.png')
    plt.clf()


### Training negative twitter model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(neg_prev_stocks, neg_fut_stocks, random_state=1)
reg = LinearRegression().fit(X_train, y_train)
prediction = reg.predict(X_test)
os.makedirs("./neg_linear", exist_ok = True) 

import matplotlib.pyplot as plt
step1 = [-5,-4,-3,-2,-1,0,1,2,3]
for index in range(len(X_test)): # pos: 2151 | 1184 neg:  | 
    real1 = X_test[index] + y_test[index]
    pred1 = X_test[index] + prediction[index].tolist()
    plt.plot(step1, real1, label='real data')
    plt.plot(step1, pred1, label='prediction data')
    plt.xlabel('time')
    plt.ylabel('price')
    plt.legend()
    plt.savefig('./neg_linear/pred'+str(index)+'.png')
    plt.clf()


#### LSTM code
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pos_prev_stocks, pos_fut_stocks, random_state=1)
X_train, X_test, y_train, y_test = torch.tensor(X_train), torch.tensor(X_test), torch.tensor(y_train), torch.tensor(y_test)

### Training postive twitter model
model_pos = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model_pos.parameters(), lr=0.01)
hist = np.zeros(num_epochs)
lstm = []
for t in range(num_epochs):
    y_train_pred = model_pos(X_train)
    loss = criterion(y_train_pred, y_train)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
torch.save(model_pos.state_dict(), "./pos.pth")

prediction = model_pos(X_test)
os.makedirs("./pos", exist_ok = True) 

import matplotlib.pyplot as plt
step1 = [-5,-4,-3,-2,-1,0,1,2,3]
for index in range(len(X_test)): # pos: 2151 | 1184 neg:  | 
	real1 = X_test[index].tolist() + y_test[index].tolist()
	pred1 = X_test[index].tolist() + prediction[index].tolist()
	plt.plot(step1, real1, label='real data')
	plt.plot(step1, pred1, label='prediction data')
	plt.xlabel('time')
	plt.ylabel('price')
	plt.legend()
	plt.savefig('./pos/pred'+str(index)+'.png')
	plt.clf()


### Training neutral twitter model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(neu_prev_stocks, neu_fut_stocks, random_state=1)
X_train, X_test, y_train, y_test = torch.tensor(X_train), torch.tensor(X_test), torch.tensor(y_train), torch.tensor(y_test)

model_neu = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model_neu.parameters(), lr=0.01)
hist = np.zeros(num_epochs)
lstm = []
for t in range(num_epochs):
    y_train_pred = model_pos(X_train)
    loss = criterion(y_train_pred, y_train)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
torch.save(model_neu.state_dict(), "./neu.pth")

prediction = model_neu(X_test)
os.makedirs("./neu", exist_ok = True) 

import matplotlib.pyplot as plt
step1 = [-5,-4,-3,-2,-1,0,1,2,3]
for index in range(len(X_test)): # pos: 2151 | 1184 neg:  | 
	real1 = X_test[index].tolist() + y_test[index].tolist()
	pred1 = X_test[index].tolist() + prediction[index].tolist()
	plt.plot(step1, real1, label='real data')
	plt.plot(step1, pred1, label='prediction data')
	plt.xlabel('time')
	plt.ylabel('price')
	plt.legend()
	plt.savefig('./neu/pred'+str(index)+'.png')
	plt.clf()


### Training negative twitter model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(neg_prev_stocks, neg_fut_stocks, random_state=1)
X_train, X_test, y_train, y_test = torch.tensor(X_train), torch.tensor(X_test), torch.tensor(y_train), torch.tensor(y_test)

model_neg = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model_neg.parameters(), lr=0.01)
hist = np.zeros(num_epochs)
lstm = []
for t in range(num_epochs):
    y_train_pred = model_pos(X_train)
    loss = criterion(y_train_pred, y_train)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
torch.save(model_neg.state_dict(), "./neg.pth")
prediction = model_neg(X_test)

os.makedirs("./neg", exist_ok = True) 
import matplotlib.pyplot as plt
step1 = [-5,-4,-3,-2,-1,0,1,2,3]
for index in range(len(X_test)): # pos: 2151 | 1184 neg:  | 
	real1 = X_test[index].tolist() + y_test[index].tolist()
	pred1 = X_test[index].tolist() + prediction[index].tolist()
	plt.plot(step1, real1, label='real data')
	plt.plot(step1, pred1, label='prediction data')
	plt.xlabel('time')
	plt.ylabel('price')
	plt.legend()
	plt.savefig('./neg/pred'+str(index)+'.png')
	plt.clf()
