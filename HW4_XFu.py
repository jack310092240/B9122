# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:27:38 2019

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


with open('stock_monthly1.pkl', 'rb') as f:
    file1 = pickle.load(f)
    
with open('.\output\stock_annual.pkl', 'rb') as f:
    file2 = pickle.load(f)

'''
'date'
'dlret'
'dlretx': 
'exchcd': Exchange Code	Exchange Code: 1 = NYSE, 2 = NYSE American, 3 = NASDAQ
'naics': industry code
'permco': CRSP Permanent Company Identifier
'prc': Price
'ret': return
'shrcd': share code
'shrout': share outstanding
'siccd': SIC Code	Standard Industrial Classification Code
'ticker'
'rankyear'
'retadj': adjusted return
'me': Market Equity (ME): shares*price
'mesum_permco': company ME, sum of ME over all shareclasses for one company 
'mesum'
'melag': lag of ME
'''
# question 2
## first step: compute break point
from fire_pytools.portools.find_breakpoints import *

file2 = file2.dropna(subset = ['mesum_dec','mesum_june'],how = 'any')
adata_me = file2.dropna(subset = ['mesum_dec','mesum_june'],how = 'any')
me_bp = find_breakpoints(data = adata_me,quantiles = {'mesum_dec':[0.5]},id_variables = ['rankyear', 'permno', 'exchcd'], exch_cd = [1])

adata_beme = file2[file2.be>0]
adata_beme = file2.dropna(subset = ['beme'],how = 'any')
beme_bp = find_breakpoints(data = adata_me,quantiles = {'beme':[0.3,0.7]},id_variables = ['rankyear', 'permno', 'exchcd'], exch_cd = [1])

adata_opbe = file2[file2.be>0]
adata_opbe = file2.dropna(subset = ['opbe'],how = 'any')
opbe_bp = find_breakpoints(data = adata_me,quantiles = {'opbe':[0.3,0.7]},id_variables = ['rankyear', 'permno', 'exchcd'], exch_cd = [1])

adata_inv = file2.dropna(subset = ['inv_gvkey'],how = 'any')
inv_bp = find_breakpoints(data = adata_me,quantiles = {'inv_gvkey':[0.3,0.7]},id_variables = ['rankyear', 'permno', 'exchcd'], exch_cd = [1])

adata_mom = file1.dropna(subset = ['CR'],how = 'any')
mom_bp = find_breakpoints(data = file1,quantiles = {'CR':[0.3,0.7]},id_variables = ['date', 'permno', 'exchcd'], exch_cd = [1])

# second step: sort the portfolio
list1 = [me_bp,beme_bp,opbe_bp,inv_bp]
list2 = ['mesum_dec','beme','opbe','inv_gvkey']
list3 = [[0.5],[0.3,0.7],[0.3,0.7],[0.3,0.7]]

port_me = sort_portfolios(data=file2,
                   quantiles={'mesum_dec':[0.5]},
                   id_variables=['rankyear', 'permno', 'exch_cd'],
                   breakpoints={'mesum_dec':me_bp})

port_beme = sort_portfolios(data=file2,
                   quantiles={'beme':[0.3,0.7]},
                   id_variables=['rankyear', 'permno', 'exch_cd'],
                   breakpoints={'beme':beme_bp})

port_opbe = sort_portfolios(data=file2,
                   quantiles={'opbe':[0.3,0.7]},
                   id_variables=['rankyear', 'permno', 'exch_cd'],
                   breakpoints={'opbe':opbe_bp})

port_inv = sort_portfolios(data=file2,
                   quantiles={'inv_gvkey':[0.3,0.7]},
                   id_variables=['rankyear', 'permno', 'exch_cd'],
                   breakpoints={'inv_gvkey':inv_bp})

port_mom = sort_portfolios(data=file1,
                   quantiles={'CR':[0.3,0.7]},
                   id_variables=['date', 'permno', 'exch_cd'],
                   breakpoints={'CR':mom_bp})

from fire_pytools.portools.sort_portfolios import *
## Now we merge our dataset
df_merge = pd.merge(file1,
                 port_me[['rankyear', 'permno','mesum_decportfolio']],
                 on=['rankyear','permno'],how = 'left')

df_merge = pd.merge(df_merge,
                 port_beme[['rankyear', 'permno','bemeportfolio']],
                 on=['rankyear','permno'],how = 'left')

df_merge = pd.merge(df_merge,
                 port_opbe[['rankyear', 'permno','opbeportfolio']],
                 on=['rankyear','permno'],how = 'left')

df_merge = pd.merge(df_merge,
                 port_inv [['rankyear', 'permno','inv_gvkeyportfolio']],
                 on=['rankyear','permno'],how = 'left')

df_merge = pd.merge(df_merge,
                 port_mom[['date', 'permno','CRportfolio']],
                 on=['date','permno'],how = 'left')


dff = df_merge[['date','rankyear','permno','retadj','melag', 'mesum_decportfolio','bemeportfolio','opbeportfolio','inv_gvkeyportfolio','CRportfolio']]


'''
Here's the formula in Fama's web:

SMB = 1/3 ( SMB(B/M) + SMB(OP) + SMB(INV) ).

HML (High Minus Low) is the average return on the two value portfolios minus the average return on the two growth portfolios,
HML = 1/2 (Small Value + Big Value) - 1/2 (Small Growth + Big Growth).	 
 	 	 
RMW (Robust Minus Weak) is the average return on the two robust operating profitability portfolios minus the average return on the two weak operating profitability portfolios,
RMW = 1/2 (Small Robust + Big Robust)- 1/2 (Small Weak + Big Weak).	 
 	 	 
CMA (Conservative Minus Aggressive) is the average return on the two conservative investment portfolios minus the average return on the two aggressive investment portfolios,
CMA =1/2 (Small Conservative + Big Conservative)- 1/2 (Small Aggressive + Big Aggressive).	 
'''

# now we compute the factor and compare with the Fama's number
# compute HML
dff = dff.dropna()
dff = dff.sort_values(['mesum_decportfolio','bemeportfolio'])
HML=dff.groupby(['date','mesum_decportfolio','bemeportfolio'])['retadj','melag'].apply(lambda x: np.average(x.retadj, weights=x.melag))
HML.to_frame()
HML=HML.reset_index()

HML = HML.rename({0 : 'return'},axis = 'columns')
HML = HML.groupby(['date'])['return'].apply(lambda t: -(t.iloc[0]+t.iloc[3]-(t.iloc[2]+t.iloc[5]))/2)
HML=HML.reset_index()
HML['date'] = HML['date'].apply(lambda x:str(x)[0:7].replace('-',''))

factor= pd.read_csv('F-F_Research_Data_5_Factors_2x3.csv', error_bad_lines=False)
HML_ff = factor[['month','HML']][:-1]
HML_ff['month'] = HML_ff['month'].astype(int)
HML['date'] =  HML['date'].astype(int) 

hml_merge = pd.merge(HML,
                 HML_ff[['month', 'HML']],
                 left_on=['date'],
                 right_on = ['month'])


print("************ corr for HML **************")
print(np.corrcoef(hml_merge['return'],hml_merge['HML']))


# compute RMW
RMW=dff.groupby(['date','mesum_decportfolio','opbeportfolio'])['retadj','melag'].apply(lambda x: np.average(x.retadj, weights=x.melag))
RMW.to_frame()
RMW=RMW.reset_index()

RMW = RMW.rename({0 : 'return'},axis = 'columns')
RMW = RMW.groupby(['date'])['return'].apply(lambda t: -(t.iloc[0]+t.iloc[3]-(t.iloc[2]+t.iloc[5]))/2)
RMW=RMW.reset_index()
RMW['date'] = RMW['date'].apply(lambda x:str(x)[0:7].replace('-',''))

factor= pd.read_csv('F-F_Research_Data_5_Factors_2x3.csv', error_bad_lines=False)
RMW_ff = factor[['month','RMW']][:-1]
RMW_ff['month'] = RMW_ff['month'].astype(int)
RMW['date'] =  RMW['date'].astype(int) 

rmw_merge = pd.merge(RMW,
                 RMW_ff[['month', 'RMW']],
                 left_on=['date'],
                 right_on = ['month'])


print("************ corr for RMW **************")
print(np.corrcoef(rmw_merge['return'],rmw_merge['RMW']))


# compute CMA
CMA=dff.groupby(['date','mesum_decportfolio','inv_gvkeyportfolio'])['retadj','melag'].apply(lambda x: np.average(x.retadj, weights=x.melag))
CMA.to_frame()
CMA=CMA.reset_index()

CMA = CMA.rename({0 : 'return'},axis = 'columns')
CMA= CMA.groupby(['date'])['return'].apply(lambda t: (t.iloc[0]+t.iloc[3]-(t.iloc[2]+t.iloc[5]))/2)
CMA=CMA.reset_index()
CMA['date'] = CMA['date'].apply(lambda x:str(x)[0:7].replace('-',''))

factor= pd.read_csv('F-F_Research_Data_5_Factors_2x3.csv', error_bad_lines=False)
CMA_ff = factor[['month','CMA']][:-1]
CMA_ff['month'] = CMA_ff['month'].astype(int)
CMA['date'] =  CMA['date'].astype(int) 

cma_merge = pd.merge(CMA,
                 CMA_ff[['month', 'CMA']],
                 left_on=['date'],
                 right_on = ['month'])


print("************ corr for CMA **************")
print(np.corrcoef(cma_merge['return'],cma_merge['CMA']))


# compute SMB
CMA1=dff.groupby(['date','mesum_decportfolio','inv_gvkeyportfolio'])['retadj','melag'].apply(lambda x: np.average(x.retadj, weights=x.melag))
CMA1.to_frame()
CMA1=CMA1.reset_index()

CMA1 = CMA1.rename({0 : 'return'},axis = 'columns')
CMA1= CMA1.groupby(['date'])['return'].apply(lambda t: (t.iloc[0]+t.iloc[1]+t.iloc[2]-(t.iloc[3]+t.iloc[4]+t.iloc[5]))/3)


HML1=dff.groupby(['date','mesum_decportfolio','bemeportfolio'])['retadj','melag'].apply(lambda x: np.average(x.retadj, weights=x.melag))
HML1.to_frame()
HML1=HML1.reset_index()

HML1 = HML1.rename({0 : 'return'},axis = 'columns')
HML1 = HML1.groupby(['date'])['return'].apply(lambda t: (t.iloc[0]+t.iloc[1]+t.iloc[2]-(t.iloc[3]+t.iloc[4]+t.iloc[5]))/3)


RMW1=dff.groupby(['date','mesum_decportfolio','opbeportfolio'])['retadj','melag'].apply(lambda x: np.average(x.retadj, weights=x.melag))
RMW1.to_frame()
RMW1=RMW1.reset_index()

RMW1 = RMW1.rename({0 : 'return'},axis = 'columns')
RMW1 = RMW1.groupby(['date'])['return'].apply(lambda t: (t.iloc[0]+t.iloc[1]+t.iloc[2]-(t.iloc[3]+t.iloc[4]+t.iloc[5]))/3)

SMB = (1/3)*(CMA1+HML1+RMW1)
SMB=SMB.reset_index()
SMB['date'] = SMB['date'].apply(lambda x:str(x)[0:7].replace('-',''))

factor= pd.read_csv('F-F_Research_Data_5_Factors_2x3.csv', error_bad_lines=False)
SMB_ff = factor[['month','SMB']][:-1]
SMB_ff['month'] = SMB_ff['month'].astype(int)
SMB['date'] =  SMB['date'].astype(int) 

smb_merge = pd.merge(SMB,
                 SMB_ff[['month', 'SMB']],
                 left_on=['date'],
                 right_on = ['month'])


print("************ corr for SMB **************")
print(np.corrcoef(smb_merge['return'],smb_merge['SMB']))

MOM=dff.groupby(['date','mesum_decportfolio','CRportfolio'])['retadj','melag'].apply(lambda x: np.average(x.retadj, weights=x.melag))
MOM.to_frame()
MOM=MOM.reset_index()

MOM = MOM.rename({0 : 'return'},axis = 'columns')
MOM= MOM.groupby(['date'])['return'].apply(lambda t: -(t.iloc[0]+t.iloc[3]-(t.iloc[2]+t.iloc[5]))/2)
MOM=MOM.reset_index()
MOM['date'] = MOM['date'].apply(lambda x:str(x)[0:7].replace('-',''))

factor = pd.read_csv('F-F_Momentum_Factor.csv', error_bad_lines=False)
MOM_ff = factor[['month','Mom   ']][:-1]
MOM_ff['month'] = MOM_ff['month'].astype(int)
MOM['date'] = MOM['date'].astype(int) 

mom_merge = pd.merge(MOM,
                 MOM_ff[['month', 'Mom   ']],
                 left_on=['date'],
                 right_on = ['month'])

print("************ corr for Mom **************")
print(np.corrcoef(mom_merge['return'],mom_merge['Mom   ']))

hml_cum = hml_merge['return'].cumsum(axis = 0)
rmw_cum = rmw_merge['return'].cumsum(axis = 0)
cma_cum = cma_merge['return'].cumsum(axis = 0)
smb_cum = smb_merge['return'].cumsum(axis = 0)
mom_cum = mom_merge['return'].cumsum(axis = 0)

plt.plot(smb_merge['date']/100,hml_cum)
plt.plot(smb_merge['date']/100,rmw_cum)
plt.plot(smb_merge['date']/100,cma_cum)
plt.plot(smb_merge['date']/100,smb_cum)













