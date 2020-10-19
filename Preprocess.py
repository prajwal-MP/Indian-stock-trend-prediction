import pandas as pd
import numpy as np
from datetime import datetime
filename = 'stock-market-india\StockMarketData_2020-02-14.h5'
Data = pd.HDFStore(filename,'r')
print(Data.keys())

keys = ['CANBK__EQ__NSE__NSE__MINUTE','AUBANK__EQ__NSE__NSE__MINUTE','AXISBANK__EQ__NSE__NSE__MINUTE','BANKINDIA__EQ__NSE__NSE__MINUTE','YESBANK__EQ__NSE__NSE__MINUTE']
for i in keys:
    BNK = Data[i]
    BNK['Year'] = [int(j[:4]) for j in BNK.index.values.astype(str)]
    BNK['Month'] = [int(j[5:7]) for j in BNK.index.values.astype(str)]
    BNK['Date'] = [int(j[8:10]) for j in BNK.index.values.astype(str)]
    BNK['Hour'] = [int(j[11:13]) for j in BNK.index.values.astype(str)]
    BNK['Minute'] = [int(j[14:16]) for j in BNK.index.values.astype(str)]
    BNK.to_csv(str(i)+'.csv')    
Data.close()

