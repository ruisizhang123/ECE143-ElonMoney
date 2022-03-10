import pandas as pd
btc=pd.read_csv('BTC.csv')
etr=pd.read_csv('ETR.csv')
doge=pd.read_csv('DOGE.csv')
twitter=pd.read_csv('data.csv')
twitter['Company']=0
twitter['price-5']=0
twitter['price-4']=0
twitter['price-3']=0
twitter['price-2']=0
twitter['price-1']=0
twitter['price0']=0
twitter['price1']=0
twitter['price2']=0
twitter['price3']=0

ddate=[]
edate=[]
bdate=[]
dindex=[]
bindex=[]
eindex=[]

"讀出twitter內所有和加密或幣有關的日期"
for i in range(len(twitter)):
    if (twitter['Class'][i])=="DogeCoin" :
        ddate.append(twitter['Time'][i])
        dindex.append(i)
    if (twitter['Class'][i]) == "Ethereum":
        edate.append(twitter['Time'][i])
        eindex.append(i)
    if (twitter['Class'][i]) == "BitCoin":
        bdate.append(twitter['Time'][i])
        bindex.append(i)

"把twitter的格式改成加密或幣的。股票的不一定要這行"
for i in range(len(ddate)):
    year=ddate[i].split('/')[0]
    month=ddate[i].split('/')[1]
    date=ddate[i].split('/')[2]
    ddate[i]=year+"-"+month+"-"+date
for i in range(len(edate)):
    year=edate[i].split('/')[0]
    month=edate[i].split('/')[1]
    date=edate[i].split('/')[2]
    edate[i]=year+"-"+month+"-"+date
for i in range(len(bdate)):
    year=bdate[i].split('/')[0]
    month=bdate[i].split('/')[1]
    date=bdate[i].split('/')[2]
    bdate[i]=year+"-"+month+"-"+date

"找到對應的日期，把價錢填進去"
for i in range(len(ddate)):
    for j in range(len(doge)):
        if str(doge['Date'][j])==str(ddate[i]):
            price_m5 = (doge['Close'][j - 5])
            price_m4 = (doge['Close'][j - 4])
            price_m3 = (doge['Close'][j - 3])
            price_m2 = (doge['Close'][j - 2])
            price_m1=(doge['Close'][j-1])
            price_0=(doge['Close'][j])
            price_1=(doge['Close'][j+1])
            price_2=(doge['Close'][j+2])
            price_3=(doge['Close'][j+3])
            twitter['Company'][dindex[i]] = 'DogeCoin'
            twitter['price-5'][dindex[i]] = price_m5
            twitter['price-4'][dindex[i]] = price_m4
            twitter['price-3'][dindex[i]] = price_m3
            twitter['price-2'][dindex[i]] = price_m2
            twitter['price-1'][dindex[i]] = price_m1
            twitter['price0'][dindex[i]] = price_0
            twitter['price1'][dindex[i]]= price_1
            twitter['price2'][dindex[i]]= price_2
            twitter['price3'][dindex[i]]= price_3


for i in range(len(edate)):
    for j in range(len(etr)):
        if str(etr['Date'][j])==str(edate[i]):
            price_m5 = (etr['Close'][j - 5])
            price_m4 = (etr['Close'][j - 4])
            price_m3 = (etr['Close'][j - 3])
            price_m2 = (etr['Close'][j - 2])
            price_m1=(etr['Close'][j-1])
            price_0=(etr['Close'][j])
            price_1=(etr['Close'][j+1])
            price_2=(etr['Close'][j+2])
            price_3=(etr['Close'][j+3])
            twitter['Company'][eindex[i]] = 'Ethereum'
            twitter['price-5'][eindex[i]] = price_m5
            twitter['price-4'][eindex[i]] = price_m4
            twitter['price-3'][eindex[i]] = price_m3
            twitter['price-2'][eindex[i]] = price_m2
            twitter['price-1'][eindex[i]] = price_m1
            twitter['price0'][eindex[i]] = price_0
            twitter['price1'][eindex[i]] = price_1
            twitter['price2'][eindex[i]] = price_2
            twitter['price3'][eindex[i]] = price_3



for i in range(len(bdate)):
    for j in range(len(btc)):
        if str(btc['Date'][j])==str(bdate[i]):
            price_m5 = (btc['Closing Price (USD)'][j - 5])
            price_m4 = (btc['Closing Price (USD)'][j - 4])
            price_m3 = (btc['Closing Price (USD)'][j - 3])
            price_m2 = (btc['Closing Price (USD)'][j - 2])
            price_m1=(btc['Closing Price (USD)'][j-1])
            price_0=(btc['Closing Price (USD)'][j])
            price_1=(btc['Closing Price (USD)'][j+1])
            price_2=(btc['Closing Price (USD)'][j+2])
            price_3=(btc['Closing Price (USD)'][j+3])
            twitter['Company'][bindex[i]] = 'BitCoin'
            twitter['price-5'][bindex[i]] = price_m5
            twitter['price-4'][bindex[i]] = price_m4
            twitter['price-3'][bindex[i]] = price_m3
            twitter['price-2'][bindex[i]] = price_m2
            twitter['price-1'][bindex[i]] = price_m1
            twitter['price0'][bindex[i]] = price_0
            twitter['price1'][bindex[i]] = price_1
            twitter['price2'][bindex[i]] = price_2
            twitter['price3'][bindex[i]] = price_3

twitter.to_csv("cryptodata.csv",index=False)