# -*- coding: utf-8 -*-
"""
Big data HW 6

*** instructions ***

In homework #5, we revised the methodology introduced by Fama and French (1993 and 1996) to construct factor-portfolios.

After Fama and French seminal work, hundreds of other characteristics have been shown to help to explain the cross-section of returns. Many call this phenomenon "zoo of factors", after Jonh Cochrane's 2011 presidential address. 

The objective of this homework is to develop a critical view of  "new factors". From the list of the of 313 papers refereed by Harvey, Liu, Zhu (RFS, 2018) in "... and the and the Cross-Section of Expected Returns" Table 6, choose one characteristic that has been considered an anomaly and perform the following tasks:

1. Update the import code to construct the characteristic chosen;

2. Calculate the characteristic sorted long-short portfolio (like in the original paper);

3. Calculate the three statistics in the paper's original sample and in the full sample: Sharpe-ratio, 3-factor alpha (alpha with respect to MktRF, SMB and HML) and 5-factor + MOM alpha (alpha with respect to MktRF, SMB, HML, RMW and CMA).  

(4.) Notice if the original paper you have chosen use value-weighted returns or equal-weighted. In the case of equal-weight, calculate also the value-weighted returns.

Submit an ipython notebook html with:

Paper reference, a table with the papers original results, your results and one paragraph of discussion about the differences in your results and the original paper.

For reference, submit also your python code and your git log. 

@author: Xiaoyong Fu
"""

import pandas as pd
import numpy as np
import datetime
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt
import pickle
from fire_pytools.import_wrds.crsp_sf import *
from fire_pytools.utils.post_event_nan import *
from fire_pytools.portools.sort_portfolios import *

import statsmodels.formula.api as sm
from statsmodels.api import add_constant
import wrds
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
from time import strptime, strftime








db = wrds.Connection(wrds_username='xf96') # make sure to configure wrds connector before hand.

fund_table = 'funda'

varlist = ['gvkey','at','csho','prcc_f','ni']

#sich h refers to historical

query = """SELECT gvkey, datadate, {}
           FROM compd.{}
           WHERE datafmt = 'STD'
           AND popsrc = 'D'
           AND indfmt = 'INDL'
           AND consol = 'C'
           AND fyear>=1980;""".format(", ".join(varlist), fund_table)

compa = db.raw_sql(query, date_cols=['datadate'])
del(fund_table, varlist, query)
print(compa.head())



import pickle

with open(r'C:\Users\fuxia\Desktop\big data\HW4\HW4\HW4 big data\output\stock_annual.pkl', 'rb') as f:
    data = pickle.load(f)


#compa=compa.drop(compa.columns[2], axis=1)


compa.columns = ['gvkey','datadate','dgvkey','at','csho','prcc_f','ni']


dataPE=data.merge(compa, on=['gvkey','at'], how='inner')
dataPE['pe']=dataPE['csho'] * dataPE['prcc_f'] / dataPE['ni']

# delete nan and inf
dataPE = dataPE.replace([np.inf, -np.inf], np.nan).dropna()
# only take value of pe > 0 
dataPE = dataPE[dataPE.pe>0]

# sort and get portfolio labels as usual
port_pe = sort_portfolios(data=dataPE,
                   quantiles={'pe':[0.2,0.4,0.6,0.8]},
                   id_variables=['rankyear', 'permno', 'exch_cd'])


with open('stock_monthly1.pkl', 'rb') as f:
    file1 = pickle.load(f)
with open('.\output\stock_annual.pkl', 'rb') as f:
    file2 = pickle.load(f)

# add the column 'permno' in to the dataframe 
file_merge = pd.merge(file1,
                 file2[['rankyear', 'permno','permco']],
                 on=['rankyear','permco'],how = 'left')
file_merge = pd.merge(file_merge,
                 dataPE[['rankyear', 'permno','pe']],
                 on=['rankyear','permno'],how = 'left')

df_interest = file_merge[['date','rankyear','permno','retadj','melag','pe']]

df_merge = pd.merge(df_interest,
                 port_pe[['rankyear', 'permno','peportfolio']],
                 on=['rankyear','permno'],how = 'left')

# finally we get the dataset
dff = df_merge.dropna()
print(list(dff.columns.values))


# import fama 5 factor + MOM
factor= pd.read_csv('F-F_Research_Data_5_Factors_2x3.csv', error_bad_lines=False)
factor['date'] = factor['date'].astype(int)

# For portfolio A
df1 = dff[dff.peportfolio == 'pe1'] 

# compute the portfolio weighted average return
ret1=df1.groupby(['date'])['retadj','melag'].apply(lambda x: np.average(x.retadj, weights=x.melag))
ret1.to_frame()
ret1=ret1.reset_index()
ret1 = ret1.rename({0 : 'return'},axis = 'columns')

# to make the form of 'date' to match in both dataset
ret1['date'] = ret1['date'].apply(lambda x:str(x)[0:7].replace('-','')).astype(int)

# match datasets so that we could run a regression
reg1 = pd.merge(factor,
                 ret1[['date', 'return']],
                 on=['date'],how = 'left')

reg1 = reg1.dropna()
print(list(reg1.columns.values))

# sharpe ratio
reg1['Sharpe'] = (reg1['return']-reg1['Rf'])/(reg1['return'].std()*100)

# 3 factor regression
y=reg1['return']
x=reg1[['Mkt', 'SMB', 'HML']]
x = add_constant(x)
result1_3factor = sm.OLS(y,x).fit()
alpha1_3factor = np.mean(result1_3factor.resid[:-1])

# 5 factor + MOM regression
y=reg1['return']
x=reg1[['Mkt', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']]
x = add_constant(x)
result1_5factor = sm.OLS(y,x).fit()
alpha1_5factor = np.mean(result1_5factor.resid[:-1])

print("****** Portfolio A *********")
print("Median PE ratio is: ")
print(np.median(df1['pe']))
print("Median Sharpe ratio is: ")
print(np.median(reg1['Sharpe']))
print("Average alpha is: ")
print(alpha1_3factor, alpha1_5factor)






# For portfolio E
df5 = dff[dff.peportfolio == 'pe5'] 

# compute the portfolio weighted average return
ret5=df5.groupby(['date'])['retadj','melag'].apply(lambda x: np.average(x.retadj, weights=x.melag))
ret5.to_frame()
ret5=ret5.reset_index()
ret5 = ret5.rename({0 : 'return'},axis = 'columns')

# to make the form of 'date' to match in both dataset
ret5['date'] = ret5['date'].apply(lambda x:str(x)[0:7].replace('-','')).astype(int)

# match datasets so that we could run a regression
reg5 = pd.merge(factor,
                 ret5[['date', 'return']],
                 on=['date'],how = 'left')

reg5 = reg5.dropna()
print(list(reg5.columns.values))

# sharpe ratio
reg5['Sharpe'] = (reg5['return']-reg5['Rf'])/(reg5['return'].std()*100)

# 3 factor regression
y=reg5['return']
x=reg5[['Mkt', 'SMB', 'HML']]
x = add_constant(x)
result5_3factor = sm.OLS(y,x).fit()
alpha5_3factor = np.mean(result5_3factor.resid[:-1])

# 5 factor + MOM regression
y=reg5['return']
x=reg5[['Mkt', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']]
x = add_constant(x)
result5_5factor = sm.OLS(y,x).fit()
alpha5_5factor = np.mean(result5_5factor.resid[:-1])

print("****** Portfolio E *********")
print("Median PE ratio is: ")
print(np.median(df5['pe']))
print("Median Sharpe ratio is: ")
print(np.median(reg5['Sharpe']))
print("Average alpha is: ")
print(alpha5_3factor, alpha5_5factor)
