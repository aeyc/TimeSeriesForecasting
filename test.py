#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:32:44 2020

@author: Ayca
"""
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
#%% Methods
def season_labels(df):
    #tmp = copy.deepcopy(df)
    winter = [1,2,12]
    spring = [3,4,5]
    summer = [6,7,8]
    autumn = [9,10,11]
    lst =[]
    for i in range(0,len(df)):
        if df.iloc[i].Month in winter:
            lst.append("Winter")
        elif df.iloc[i].Month in spring:
            lst.append("Spring")
        elif df.iloc[i].Month in summer:
            lst.append("Summer")
        elif df.iloc[i].Month in autumn:
            lst.append( "Autumn")
    return lst

    
#%%
#Read csv files
#aep = pd.read_csv("AEP_hourly.csv",names = ["Datetime","MW"],skiprows=1)
#aep['Name'] = "AEP"
#aep["Date"] = pd.to_datetime(aep["Datetime"], errors='coerce').dt.date
#aep["Time"] = pd.to_datetime(aep["Datetime"], errors='coerce').dt.time
#aep = aep.drop(['Datetime'], axis=1)
#
#comed = pd.read_csv("COMED_hourly.csv",names = ["Datetime","MW"],skiprows=1)
#comed['Name'] = "COMED"
#comed["Date"] = pd.to_datetime(comed["Datetime"]).dt.date
#comed["Time"] = pd.to_datetime(comed["Datetime"]).dt.time
#comed = comed.drop(['Datetime'], axis=1)
#
#dayton = pd.read_csv("DAYTON_hourly.csv",names = ["Datetime","MW"],skiprows=1)
#dayton['Name'] = "DAYTON"
#dayton["Date"] = pd.to_datetime(dayton["Datetime"]).dt.date
#dayton["Time"] = pd.to_datetime(dayton["Datetime"]).dt.time
#dayton = dayton.drop(['Datetime'], axis=1)
#
#deok = pd.read_csv("DEOK_hourly.csv",names = ["Datetime","MW"],skiprows=1)
#deok['Name'] = "DEOK"
#deok["Date"] = pd.to_datetime(deok["Datetime"]).dt.date
#deok["Time"] = pd.to_datetime(deok["Datetime"]).dt.time
#deok = deok.drop(['Datetime'], axis=1)
#
#dom = pd.read_csv("DOM_hourly.csv",names = ["Datetime","MW"],skiprows=1)
#dom['Name'] = "DOM"
#dom["Date"] = pd.to_datetime(dom["Datetime"]).dt.date
#dom["Time"] = pd.to_datetime(dom["Datetime"]).dt.time
#dom = dom.drop(['Datetime'], axis=1)

ekpc = pd.read_csv("EKPC_hourly.csv",names = ["Datetime","MW"],skiprows=1)
ekpc['Datetime'] = pd.to_datetime(ekpc['Datetime'])
#ekpc['Name'] = "EKPC"
#ekpc["Date"] = pd.to_datetime(ekpc["Datetime"]).dt.date
#ekpc["Time"] = pd.to_datetime(ekpc["Datetime"]).dt.time
#ekpc = ekpc.drop(['Datetime'], axis=1)
#ekpc['Month'] = [ ekpc.iloc[i].Date.month for i in range(0,len(ekpc)) ]
#ekpc['Day'] = [ ekpc.iloc[i].Date.month for i in range(0,len(ekpc)) ]
#ekpc['Season'] = season_labels(ekpc)

    

#fe = pd.read_csv("FE_hourly.csv",names = ["Datetime","MW"],skiprows=1)
#fe['Name'] = "FE"
#fe["Date"] = pd.to_datetime(fe["Datetime"]).dt.date
#fe["Time"] = pd.to_datetime(fe["Datetime"]).dt.time
#fe = fe.drop(['Datetime'], axis=1)
#
#ni = pd.read_csv("NI_hourly.csv",names = ["Datetime","MW"],skiprows=1)
#ni['Name'] = "NI"
#ni["Date"] = pd.to_datetime(ni["Datetime"]).dt.date
#ni["Time"] = pd.to_datetime(ni["Datetime"]).dt.time
#ni = ni.drop(['Datetime'], axis=1)
#
#pjme = pd.read_csv("PJME_hourly.csv",names = ["Datetime","MW"],skiprows=1)
#pjme['Name'] = "PJME"
#pjme["Date"] = pd.to_datetime(pjme["Datetime"]).dt.date
#pjme["Time"] = pd.to_datetime(pjme["Datetime"]).dt.time
#pjme = pjme.drop(['Datetime'], axis=1)
#
#pjmw = pd.read_csv("PJMW_hourly.csv",names = ["Datetime","MW"],skiprows=1)
#pjmw['Name'] = "PJMW"
#pjmw["Date"] = pd.to_datetime(pjmw["Datetime"]).dt.date
#pjmw["Time"] = pd.to_datetime(pjmw["Datetime"]).dt.time
#pjmw = pjmw.drop(['Datetime'], axis=1)

#df = pd.concat([aep,comed,dayton,deok,dom,ekpc,fe,ni,pjme,pjmw] )
#df['Month'] = [ df.iloc[i].Date.month for i in range(0,len(df)) ]
#df['Day'] = [ df.iloc[i].Date.month for i in range(0,len(df)) ]
#df['Season'] = season_labels(df)
#names = ["AEP","COMED","DAYTON","DEOK","DOM","EKPC","FE","NI","PJME","PJMW"]

#%%

#min-max - converted csv
#print("Max of AEP:\n",aep[aep.AEP_MW == aep.AEP_MW.max()]) #AEP_MW 25695.0,Date 2008-10-20,Time 14:00:00
#print("\nMin of AEP:\n",aep[aep.AEP_MW == aep.AEP_MW.min()]) #AEP_MW  9581.0,Date 2016-10-02,Time 05:00:00

#lst = []
#lst.append(("AEP",aep[aep.MW == aep.MW.max()],aep[aep.MW == aep.MW.min()]))
#lst.append(("COMED",comed[comed.MW == comed.MW.max()],comed[comed.MW == comed.MW.min()]))
#lst.append(("DAYTON",dayton[dayton.MW == dayton.MW.max()],dayton[dayton.MW == dayton.MW.min()]))
#lst.append(("DEOK",deok[deok.MW == deok.MW.max()],deok[deok.MW == deok.MW.min()]))
#lst.append(("DOM",dom[dom.MW == dom.MW.max()],dom[dom.MW == dom.MW.min()]))
#lst.append(("EKPC",ekpc[ekpc.MW == ekpc.MW.max()],ekpc[ekpc.MW == ekpc.MW.min()]))
#lst.append(("FE",fe[fe.MW == fe.MW.max()],fe[fe.MW == fe.MW.min()]))
#lst.append(("NI",ni[ni.MW == ni.MW.max()],ni[ni.MW == ni.MW.min()]))
#lst.append(("PJME",pjme[pjme.MW == pjme.MW.max()],pjme[pjme.MW == pjme.MW.min()]))
#lst.append(("PJMW",pjmw[pjmw.MW == pjmw.MW.max()],pjmw[pjmw.MW == pjmw.MW.min()]))

#minmax = pd.read_csv("minmax.csv")
#aep["Date"] = pd.to_datetime(aep["Date"], errors='coerce').dt.date
#aep["Time"] = pd.to_datetime(aep["Date"], errors='coerce').dt.time
#minmax['Month'] = [ minmax.iloc[i].Date.month for i in range(0,len(minmax)) ]
#minmax['Day'] = [ minmax.iloc[i].Date.month for i in range(0,len(minmax)) ]
#minmax['Season'] = season_labels(minmax)
#max_df = minmax[minmax.MinMax == "max"]
#max_df = max_df.drop(['MinMax'], axis=1)
#min_df = minmax[minmax.MinMax == "min"]
#min_df = min_df.drop(['MinMax'], axis=1)
#print("Total",minmax)
#print("min",min_df)
#print("min",max_df)
#%%
print("Min",ekpc['Datetime'].min()) 
print("Min",ekpc['Datetime'].max()) 
#%%
print(ekpc.info())
print(ekpc.describe)
#%%
ekpc = ekpc.sort_values('Datetime')
print("After sort\n")
print(ekpc.head())
#%%
print("Check values\n",ekpc.isnull().sum())
#%%
ekpc = ekpc.groupby('Datetime')
ekpc = ekpc['MW'].sum().reset_index()
print("After groupby\n")
print(ekpc.head())
#%%
ekpc = ekpc.set_index('Datetime')
print("Index\n")
print(ekpc.index)
#%%
y = ekpc['MW'].resample('MS').mean()
print("Resampling\n")
print(y)
#%%
print("Resampling--2013\n")
print(y['2013':])
print("\nResampling--2014\n")
print(y['2014':])
print("\nResampling--2015\n")
print(y['2015':])
print("\nResampling--2016\n")
print(y['2016':])
print("\nResampling--2017\n")
print(y['2017':])
print("\nResampling--2018\n")
print(y['2018':])
#%%
y.plot(figsize=(15, 6))
plt.show()
#%%
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()
#%%
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
#%%
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
#OPTIMAL: ARIMA(1, 1, 1)x(1, 1, 1, 12)12 - AIC:467.7157683938848
#%%
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
#%%
results.plot_diagnostics(figsize=(16, 8))
plt.show()
#%%
pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('MW')
plt.legend()
plt.show()
#%%
y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
#The Mean Squared Error of our forecasts is 20147.61
#The Root Mean Squared Error of our forecasts is 141.94
#%%
#ekpc = season_labels(ekpc)
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split

#df_train, df_test = train_test_split(ekpc, train_size=0.8, test_size=0.2, shuffle=False)


#%%
#kNN

