import pandas as pd
TSLA = pd.read_csv('TSLA.csv')
NVDA = pd.read_csv('NVDA.csv')
GOOGL = pd.read_csv('GOOGL.csv')
MSFT = pd.read_csv('MSFT.csv')
AAPL = pd.read_csv('AAPL.csv')
ENPH = pd.read_csv('ENPH.csv')
XPEV = pd.read_csv('XPEV.csv')
GM = pd.read_csv('GM.csv')
CLNE = pd.read_csv('CLNE.csv')
ADI = pd.read_csv('ADI.csv')
NXPI = pd.read_csv('NXPI.csv')
NIO = pd.read_csv('NIO.csv')
LCID = pd.read_csv('LCID.csv')
ALB = pd.read_csv('ALB.csv')
RIVN = pd.read_csv('RIVN.csv')
LI = pd.read_csv('LI.csv')
BA = pd.read_csv('BA.csv')
AJRD = pd.read_csv('AJRD.csv')
MAXR = pd.read_csv('MAXR.csv')
SPCE = pd.read_csv('SPCE.csv')
RBLX = pd.read_csv('RBLX.csv')
FB = pd.read_csv('FB.csv')
SHOP = pd.read_csv('SHOP.csv')
MTTR = pd.read_csv('MTTR.csv')
MVIS = pd.read_csv('MVIS.csv')
VUZI = pd.read_csv('VUZI.csv')
SNAP = pd.read_csv('SNAP.csv')


data_file = pd.read_csv('newstock33.csv')

ddate=[]
dindex=[]
for i in range(len(data_file)):
    if (data_file['Class'][i]) == "Rocket":
        ddate.append(data_file['Time'][i])
        dindex.append(i)
dddate=[]
for i in range(len(ddate)):
    y=ddate[i].split('/')[0]
    m=ddate[i].split('/')[1]
    d=ddate[i].split('/')[2]
    if ddate[i].split('/')[2][0] == "0":
        d = ddate[i].split('/')[2][1]
    if ddate[i].split('/')[1][0] == "0":
        m = ddate[i].split('/')[1][1]
    dddate.append(m+"/"+d+"/"+y)
ddate=dddate

date=TSLA["Date"]
stock=TSLA["Close"]

def price(date, stock):
    for i in range(len(ddate)):
        TT = False
        for j in range(len(date)):
            if ddate[i] == date[j]:
                data_file['Company'][dindex[i]] = ['BA', 'AJRD', 'MAXR', 'SPCE']
                price_m5 = (stock['Close'][j - 5])
                price_m4 = (stock['Close'][j - 4])
                price_m3 = (stock['Close'][j - 3])
                price_m2 = (stock['Close'][j - 2])
                price_m1 = (stock['Close'][j - 1])
                price_0 = (stock['Close'][j])
                price_1 = (stock['Close'][j + 1])
                price_2 = (stock['Close'][j + 2])
                price_3 = (stock['Close'][j + 3])

                #data_file['price-5'][dindex[i]] = (price_m5)
                #data_file['price-4'][dindex[i]] = (price_m4)
                #data_file['price-3'][dindex[i]] = (price_m3)
                #data_file['price-2'][dindex[i]] = (price_m2)
                #data_file['price-1'][dindex[i]]= (price_m1)
                #data_file['price0'][dindex[i]] = (price_0)
                #data_file['price1'][dindex[i]] = (price_1)
                #data_file['price2'][dindex[i]] = (price_2)
                #data_file['price3'][dindex[i]]= (price_3)

                data_file['price-5'][dindex[i]] = str(data_file['price-5'][dindex[i]])+","+str(price_m5)
                data_file['price-4'][dindex[i]] =str (data_file['price-4'][dindex[i]])+","+str(price_m4)
                data_file['price-3'][dindex[i]] = str(data_file['price-3'][dindex[i]])+","+str(price_m3)
                data_file['price-2'][dindex[i]] = str(data_file['price-2'][dindex[i]])+","+ str(price_m2)
                data_file['price-1'][dindex[i]] = str(data_file['price-1'][dindex[i]])+","+str(price_m1)
                data_file['price0'][dindex[i]] =str (data_file['price0'][dindex[i]])+","+str(price_0)
                data_file['price1'][dindex[i]] =str (data_file['price1'][dindex[i]])+","+str(price_1)
                data_file['price2'][dindex[i]] = str(data_file['price2'][dindex[i]])+","+str(price_2)
                data_file['price3'][dindex[i]] =str (data_file['price3'][dindex[i]])+","+str(price_3)
                TT = True
                break
        if TT == False:
            twtime = []
            YY = False
            for a in range(-2, 30, 1):
                twtime=(ddate[i].split('/')[0])+"/"+str(int(ddate[i].split('/')[1])+a)+"/"+(ddate[i].split('/')[2])

                for j in range(len(date)):
                    if twtime == date[j]:

                        data_file['Company'][dindex[i]] = ['BA', 'AJRD', 'MAXR', 'SPCE']
                        price_m5 = (stock['Close'][j - 5])
                        price_m4 = (stock['Close'][j - 4])
                        price_m3 = (stock['Close'][j - 3])
                        price_m2 = (stock['Close'][j - 2])
                        price_m1 = (stock['Close'][j - 1])
                        price_0 = (stock['Close'][j])
                        price_1 = (stock['Close'][j + 1])
                        price_2 = (stock['Close'][j + 2])
                        price_3 = (stock['Close'][j + 3])

                        #data_file['price-5'][dindex[i]] = (price_m5)
                        #data_file['price-4'][dindex[i]] = (price_m4)
                        #data_file['price-3'][dindex[i]] = (price_m3)
                        #data_file['price-2'][dindex[i]] = (price_m2)
                        #data_file['price-1'][dindex[i]] = (price_m1)
                        #data_file['price0'][dindex[i]] = (price_0)
                        #data_file['price1'][dindex[i]] = (price_1)
                        #data_file['price2'][dindex[i]] = (price_2)
                        #data_file['price3'][dindex[i]] = (price_3)

                        data_file['price-5'][dindex[i]] = str(data_file['price-5'][dindex[i]]) + "," + str(price_m5)
                        data_file['price-4'][dindex[i]] = str(data_file['price-4'][dindex[i]]) + "," + str(price_m4)
                        data_file['price-3'][dindex[i]] = str(data_file['price-3'][dindex[i]]) + "," + str(price_m3)
                        data_file['price-2'][dindex[i]] = str(data_file['price-2'][dindex[i]]) + "," + str(price_m2)
                        data_file['price-1'][dindex[i]] = str(data_file['price-1'][dindex[i]]) + "," + str(price_m1)
                        data_file['price0'][dindex[i]] = str(data_file['price0'][dindex[i]]) + "," + str(price_0)
                        data_file['price1'][dindex[i]] = str(data_file['price1'][dindex[i]]) + "," + str(price_1)
                        data_file['price2'][dindex[i]] = str(data_file['price2'][dindex[i]]) + "," + str(price_2)
                        data_file['price3'][dindex[i]] = str(data_file['price3'][dindex[i]]) + "," + str(price_3)
                        YY = True
                        break
                if YY:
                    break

date=SPCE["Date"]
stock=SPCE

# 'BA', 'AJRD', 'MAXR', 'SPCE']

price(date, stock)
#print(data_file)
data_file.to_csv("newstock34.csv",index=False)