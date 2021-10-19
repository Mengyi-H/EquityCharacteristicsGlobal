
import pandas as pd
import numpy as np
import datetime as dt
import wrds
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
import datetime
import pickle as pkl
import multiprocessing as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyarrow.feather as feather
import time
from functions import *

# ###################
# # Connect to WRDS #
# ###################
# conn = wrds.Connection()


# # Download data and store locally



# fundq by country (headquarter)

# fundq = conn.raw_sql("""
#                       select *
#                       from comp.g_fundq
#                       where datadate > '01/01/1990'
#                       and loc = 'HKG'
#                       """)

# fundq = fundq.sort_values(['gvkey','datadate','iid','isin','sedol']) # order by gvkey, date, issue id, other id's
# with open('./fundq_hkg_1990.pkl', 'wb') as f:
#    pkl.dump(fundq, f)


# secd by country (headquarter)


# secd = conn.raw_sql("""
#                       select *
#                       from comp.g_secd
#                       where datadate > '01/01/1990'
#                       and loc = 'HKG'
#                       """)
# secd = secd.sort_values(['gvkey','datadate','iid','isin','sedol']) # order by gvkey, date, issue id, other id's
# with open('./secd_hkg_1990.pkl', 'wb') as f:
#     pkl.dump(secd, f)


# Read secd data

# secd = pd.read_pickle('./secd_hkg_1990.pkl')
with open('secd_hkg.feather', 'rb') as f:
    secd = feather.read_feather(f)

# data cleaning
# secd = secd[~secd['isin'].isna()]   # international id
# secd = secd[~secd['sedol'].isna()]   # international id
secd = secd[~secd['cshoc'].isna()]  # number of common shares
secd = secd[secd['tpci']=='0']      # issue tyoe code, 0 for equities 
# secd = secd[ (secd['exchg'] == 251)] # Hongkong
# secd = secd[secd['curcdd']=='HKD']  # currency
secd = secd[secd['datadate'] >= datetime.datetime.strptime('2000-01-01','%Y-%m-%d').date()]


# ### calculate me
# 
# - sort the different issues of the same firm
# - Be careful about cross-listing, like Alibaba in US and HK, SinoPetro in CN, HK, US.
# - check this later.



secd['me'] = secd['prccd']*secd['cshoc']
secd = secd.sort_values(['gvkey','datadate','me','iid']) # order by gvkey, date, issue id, other id's
secd.index = range(len(secd.index))


# calculate daily returns
secd['prc_adj'] = secd['prccd']/secd['ajexdi']
secd['prc_trfd'] = secd['prccd']/secd['ajexdi']*secd['trfd']
secd['prc_trfd_last_day'] = secd.groupby(['gvkey'])['prc_trfd'].shift(1)
secd['ret'] = secd['prc_trfd']/secd['prc_trfd_last_day']-1
# secd.columns

varlist=['gvkey', 'exchg','tpci', 'prcstd', 'loc','fic', 'iid', 'datadate', 'cshoc','conm','monthend','curcdd', 'prccd','prc_trfd','prc_trfd_last_day','ajexdi','trfd','prc_adj','me','ret']
secd = secd[varlist]

# returns
secd['ret_plus_one'] = secd['ret']+1
secd['monthid'] = secd.groupby(['gvkey']).cumsum()['monthend']
secd['cumret'] = secd[['gvkey','monthid','ret_plus_one']].groupby(['gvkey']).cumprod()['ret_plus_one']



# # Work on monthly frequency

secm = secd[secd['monthend']==1]
secm['cumret_last_month'] = secm.groupby('gvkey').shift(1)['cumret']
secm['retm'] = secm['cumret']/secm['cumret_last_month']-1




# tmp=secm.groupby(['gvkey','exchg']).nunique()['isin']
# tmp[tmp>1]
# if you remove the currency or exchange filter, you will see one gvkey links to multiple iid/isin/sedol
# now, for the China case, we have one-one-mapping between gvkey and iid (issues)


# # Read fundq

with open('fundq_hkg.feather', 'rb') as f:
    fundq = feather.read_feather(f)



# ### filters on fundq


# fundq = fundq[~fundq['isin'].isna()]   # international id
# fundq = fundq[~fundq['sedol'].isna()]   # international id
# fundq = fundq[fundq['exchg'] == 251] # hongkong
fundq = fundq[fundq['datadate'] >= datetime.datetime.strptime('2000-01-01','%Y-%m-%d').date()]


fundamental_varlist=[
    # id
    'gvkey', 'indfmt', 'consol', 'popsrc', 'datafmt','exchg', 'loc','fic', 'datadate','pdateq','fdateq','fyr',
    # varaibles we want 
    'ibq','iby',
    'seqq','txdbq','txtq','pstkq','dpy','dpq','atq',
    'cheq','actq','gdwlq','intanq','ceqq',
    'ivaoq','dlcq','dlttq','mibq','saleq','saley',
    'ltq','ppentq','revtq','cogsq',
    'rectq','acoq','apq','lcoq','loq','invtq','aoq','xintq','xsgaq','oiadpq','oancfy'
    ]
fundq = fundq[fundamental_varlist]
# fundq.head(50)




# fundq = fundq[~fundq['pdateq'].isna()] # some empty observations in fundq, you can check this with the next commented code
# fundq[fundq['gvkey']=='029530'].head(15)


fundq = fundq.sort_values(['gvkey','datadate','exchg','seqq'])




# tmp=fundq.groupby(['gvkey','exchg']).nunique()['isin']
# tmp[tmp>1]
# make sure, one gvkey-exchange has only one isin/sedol


# ### drop some observations which losses critical info




# print(fundq.shape)
# fundq = fundq[~fundq['seqq'].isna()]
# fundq = fundq[~fundq['ibq'].isna()]
# print(fundq.shape)


# ### impute some variables to zero

fundq['txdbq'] = fundq['txdbq'].fillna(0)
fundq['txtq'] = fundq['txtq'].fillna(0)
fundq['pstkq'] = fundq['pstkq'].fillna(0)

fundq['mibq'] = fundq['mibq'].fillna(0)
fundq['dlcq'] = fundq['dlcq'].fillna(0)
fundq['ivaoq'] = fundq['ivaoq'].fillna(0)
fundq['dlttq'] = fundq['dlttq'].fillna(0)


# ## Calculate fundamental information

fundq['beq'] = fundq['seqq'] + fundq['txdbq'] + fundq['txtq'] - fundq['pstkq']




# # Merge fundq(quartly) and secm(monthly)


fundq['datadate'] = pd.to_datetime(fundq['datadate'])
# join date is jdate
# quarterly fundamentals are expected to report later than the datadate
# 3 month is enough for the reporting process
# thus, we don't have forseeing-data problem
fundq['jdate'] = fundq['datadate'] + MonthEnd(3)
fundq = fundq.sort_values(['gvkey','datadate','exchg'])



secm['datadate'] = pd.to_datetime(secm['datadate'])
secm['jdate'] = secm['datadate'] + MonthEnd(0)
secm = secm.sort_values(['gvkey','datadate','exchg'])
secm = secm[['gvkey', 'exchg', 'loc', 'fic', 'iid', 'datadate', 
             'cshoc', 'prccd', 'me', 'retm', 'jdate']]


# fqsm = pd.merge(secm, fundq, how='left', on=['gvkey','jdate','exchg','loc','fic'])
fqsm = pd.merge(secm, fundq, how='right', on=['gvkey','jdate','exchg','loc','fic'])
############### actually below is fqsq #############
fqsm = fqsm.dropna(subset=['seqq'])


# # Forward Fill the Fundq info to Empty Month


# fqsm.columns


fqsm.columns = ['gvkey', 'exchg', 'loc', 'fic', 'iid', 'datadate_secm',
       'cshoc', 'prccd', 'me', 'retm', 'jdate', 'indfmt', 'consol', 'popsrc',
       'datafmt', 'datadate_fundq', 'pdateq', 'fdateq', 'fyr', 'ibq', 'iby',
       'seqq', 'txdbq', 'txtq', 'pstkq', 'dpy', 'dpq','atq', 'cheq','actq','gdwlq','intanq','ceqq',
        'ivaoq','dlcq','dlttq','mibq','saleq','saley','ltq','ppentq','revtq','cogsq',
        'rectq','acoq','apq','lcoq','loq','invtq','aoq','xintq','xsgaq','oiadpq','oancfy','beq']



fqsm = fqsm.sort_values(['gvkey','jdate']) # order by gvkey, date, issue id, other id's








fqsm['me'] = fqsm['me']/1e6 # 1e6 is one million


fqsm['bm'] = fqsm['beq']/fqsm['me']
fqsm['mb'] = fqsm['me']/fqsm['beq']


# fqsm.columns.values

# fundq.columns


# secd.columns



# fqsm.columns

########### ttm functions #########

def ttm4(series, df):
    """

    :param series: variables' name
    :param df: dataframe
    :return: ttm4
    """
    lag = pd.DataFrame()
    for i in range(1,4):
        lag['%(series)s%(lag)s' % {'series': series, 'lag': i}] = df.groupby('gvkey')['%s' % series].shift(i)
    result = df['%s' % series] + lag['%s1' % series] + lag['%s2' % series] + lag['%s3' % series]
    return result
# changes from accounting_60.py: shift(3), shift(6), shift(9)

def ttm12(series, df):
    """

    :param series: variables' name
    :param df: dataframe
    :return: ttm12
    """
    lag = pd.DataFrame()
    for i in range(1, 12):
        lag['%(series)s%(lag)s' % {'series': series, 'lag': i}] = df.groupby('gvkey')['%s' % series].shift(i)
    result = df['%s' % series] + lag['%s1' % series] + lag['%s2' % series] + lag['%s3' % series] + \
                lag['%s4' % series] + lag['%s5' % series] + lag['%s6' % series] + lag['%s7' % series] + \
                            lag['%s8' % series] + lag['%s9' % series] + lag['%s10' % series] + lag['%s11' % series]
    return result



fqsm['earnings'] = ttm4('ibq',fqsm)
# fqsm['earnings'] = fqsm['ibq'] + fqsm['ibq'].shift(3) + fqsm['ibq'].shift(6) + fqsm['ibq'].shift(9)
fqsm['ep'] = fqsm['earnings'] / fqsm['me']
fqsm['pe'] = fqsm['me'] / fqsm['earnings']

# fqsm['ep2'] = ttm4('ibq',fqsm) / fqsm['me'] * 1e6
# fqsm['pe2'] = fqsm['me']/ttm4('ibq',fqsm) / 1e6




#dpq fillna
# cp
fqsm['cf'] = ttm4('ibq',fqsm) + ttm4('dpq',fqsm)
fqsm['cp'] = fqsm['cf']/fqsm['me']


# agr
fqsm['atq_l4'] = fqsm.groupby('gvkey')['atq'].shift(4)
fqsm['agr'] = (fqsm['atq'] - fqsm['atq_l4']) / fqsm['atq_l4']



# alm
fqsm['ala'] = fqsm['cheq'] + 0.75*(fqsm['actq']-fqsm['cheq'])+\
                 0.5*(fqsm['atq']-fqsm['actq']-fqsm['gdwlq']-fqsm['intanq'])
fqsm['alm'] = fqsm['ala']/(fqsm['atq']+fqsm['me']-fqsm['ceqq'])




# ato
fqsm['noa'] = (fqsm['atq']-fqsm['cheq']-fqsm['ivaoq'])-\
                 (fqsm['atq']-fqsm['dlcq']-fqsm['dlttq']-fqsm['mibq']-fqsm['pstkq']-fqsm['ceqq'])/fqsm['atq_l4']
fqsm['noa_l4'] = fqsm.groupby(['gvkey'])['noa'].shift(4)
fqsm['ato'] = fqsm['saleq']/fqsm['noa_l4']




# cash
fqsm['cash'] = fqsm['cheq']/fqsm['atq']





# cashdebt
fqsm['ltq_l4'] = fqsm.groupby(['gvkey'])['ltq'].shift(4)
fqsm['cashdebt'] = (ttm4('ibq', fqsm) + ttm4('dpq', fqsm))/((fqsm['ltq']+fqsm['ltq_l4'])/2)



#chpm
fqsm['ibq4'] = ttm4('ibq', fqsm)
fqsm['saleq4'] = ttm4('saleq', fqsm)
fqsm['saleq4'] = np.where(fqsm['saleq4'].isnull(), fqsm['saley'], fqsm['saleq4'])
fqsm['ibq4_l1'] = fqsm.groupby(['gvkey'])['ibq4'].shift(1)
fqsm['saleq4_l1'] = fqsm.groupby(['gvkey'])['saleq4'].shift(1)
fqsm['chpm'] = (fqsm['ibq4']/fqsm['saleq4'])-(fqsm['ibq4_l1']/fqsm['saleq4_l1'])



#chtx
fqsm['txtq_l4'] = fqsm.groupby(['gvkey'])['txtq'].shift(4)
fqsm['atq_l4'] = fqsm.groupby(['gvkey'])['atq'].shift(4)
fqsm['chtx'] = (fqsm['txtq']-fqsm['txtq_l4'])/fqsm['atq_l4']



#cinvest
fqsm['ppentq_l1'] = fqsm.groupby(['gvkey'])['ppentq'].shift(1)
fqsm['ppentq_l2'] = fqsm.groupby(['gvkey'])['ppentq'].shift(2)
fqsm['ppentq_l3'] = fqsm.groupby(['gvkey'])['ppentq'].shift(3)
fqsm['ppentq_l4'] = fqsm.groupby(['gvkey'])['ppentq'].shift(4)
fqsm['saleq_l1'] = fqsm.groupby(['gvkey'])['saleq'].shift(1)
fqsm['saleq_l2'] = fqsm.groupby(['gvkey'])['saleq'].shift(2)
fqsm['saleq_l3'] = fqsm.groupby(['gvkey'])['saleq'].shift(3)

fqsm['c_temp1'] = (fqsm['ppentq_l1'] - fqsm['ppentq_l2']) / fqsm['saleq_l1']
fqsm['c_temp2'] = (fqsm['ppentq_l2'] - fqsm['ppentq_l3']) / fqsm['saleq_l2']
fqsm['c_temp3'] = (fqsm['ppentq_l3'] - fqsm['ppentq_l4']) / fqsm['saleq_l3']

fqsm['c_temp1'] = (fqsm['ppentq_l1'] - fqsm['ppentq_l2']) / 0.01
fqsm['c_temp2'] = (fqsm['ppentq_l2'] - fqsm['ppentq_l3']) / 0.01
fqsm['c_temp3'] = (fqsm['ppentq_l3'] - fqsm['ppentq_l4']) / 0.01

fqsm['cinvest'] = ((fqsm['ppentq'] - fqsm['ppentq_l1']) / fqsm['saleq'])\
                       -(fqsm[['c_temp1', 'c_temp2', 'c_temp3']].mean(axis=1))
fqsm['cinvest'] = np.where(fqsm['saleq']<=0, ((fqsm['ppentq'] - fqsm['ppentq_l1']) / 0.01)
                                -(fqsm[['c_temp1', 'c_temp2', 'c_temp3']].mean(axis=1)), fqsm['cinvest'])

fqsm = fqsm.drop(['c_temp1', 'c_temp2', 'c_temp3'], axis=1)



#depr
fqsm['depr'] = ttm4('dpq', fqsm)/fqsm['ppentq']





#gma
fqsm['revtq4'] = ttm4('revtq', fqsm)
fqsm['cogsq4'] = ttm4('cogsq', fqsm)
fqsm['atq_l4'] = fqsm.groupby(['gvkey'])['atq'].shift(4)
fqsm['gma'] = (fqsm['revtq4']-fqsm['cogsq4'])/fqsm['atq_l4']



#grltnoa
fqsm['rectq_l4'] = fqsm.groupby(['gvkey'])['rectq'].shift(4)
fqsm['acoq_l4'] = fqsm.groupby(['gvkey'])['acoq'].shift(4)
fqsm['apq_l4'] = fqsm.groupby(['gvkey'])['apq'].shift(4)
fqsm['lcoq_l4'] = fqsm.groupby(['gvkey'])['lcoq'].shift(4)
fqsm['loq_l4'] = fqsm.groupby(['gvkey'])['loq'].shift(4)
fqsm['invtq_l4'] = fqsm.groupby(['gvkey'])['invtq'].shift(4)
fqsm['ppentq_l4'] = fqsm.groupby(['gvkey'])['ppentq'].shift(4)
fqsm['atq_l4'] = fqsm.groupby(['gvkey'])['atq'].shift(4)

fqsm['grltnoa'] = ((fqsm['rectq']+fqsm['invtq']+fqsm['ppentq']+fqsm['acoq']+fqsm['intanq']+
                       fqsm['aoq']-fqsm['apq']-fqsm['lcoq']-fqsm['loq'])-
                      (fqsm['rectq_l4']+fqsm['invtq_l4']+fqsm['ppentq_l4']+fqsm['acoq_l4']-fqsm['apq_l4']-fqsm['lcoq_l4']-fqsm['loq_l4'])-\
                     (fqsm['rectq']-fqsm['rectq_l4']+fqsm['invtq']-fqsm['invtq_l4']+fqsm['acoq']-
                      (fqsm['apq']-fqsm['apq_l4']+fqsm['lcoq']-fqsm['lcoq_l4'])-
                      ttm4('dpq', fqsm)))/((fqsm['atq']+fqsm['atq_l4'])/2)



#lev
fqsm['lev'] = fqsm['ltq']/fqsm['me']

#lgr
fqsm['ltq_l4'] = fqsm.groupby(['gvkey'])['ltq'].shift(4)
fqsm['lgr'] = (fqsm['ltq']/fqsm['ltq_l4'])-1



#nincr
fqsm['ibq_l1'] = fqsm.groupby(['gvkey'])['ibq'].shift(1)
fqsm['ibq_l2'] = fqsm.groupby(['gvkey'])['ibq'].shift(2)
fqsm['ibq_l3'] = fqsm.groupby(['gvkey'])['ibq'].shift(3)
fqsm['ibq_l4'] = fqsm.groupby(['gvkey'])['ibq'].shift(4)
fqsm['ibq_l5'] = fqsm.groupby(['gvkey'])['ibq'].shift(5)
fqsm['ibq_l6'] = fqsm.groupby(['gvkey'])['ibq'].shift(6)
fqsm['ibq_l7'] = fqsm.groupby(['gvkey'])['ibq'].shift(7)
fqsm['ibq_l8'] = fqsm.groupby(['gvkey'])['ibq'].shift(8)

fqsm['nincr_temp1'] = np.where(fqsm['ibq'] > fqsm['ibq_l1'], 1, 0)
fqsm['nincr_temp2'] = np.where(fqsm['ibq_l1'] > fqsm['ibq_l2'], 1, 0)
fqsm['nincr_temp3'] = np.where(fqsm['ibq_l2'] > fqsm['ibq_l3'], 1, 0)
fqsm['nincr_temp4'] = np.where(fqsm['ibq_l3'] > fqsm['ibq_l4'], 1, 0)
fqsm['nincr_temp5'] = np.where(fqsm['ibq_l4'] > fqsm['ibq_l5'], 1, 0)
fqsm['nincr_temp6'] = np.where(fqsm['ibq_l5'] > fqsm['ibq_l6'], 1, 0)
fqsm['nincr_temp7'] = np.where(fqsm['ibq_l6'] > fqsm['ibq_l7'], 1, 0)
fqsm['nincr_temp8'] = np.where(fqsm['ibq_l7'] > fqsm['ibq_l8'], 1, 0)

fqsm['nincr'] = (fqsm['nincr_temp1']
                      + (fqsm['nincr_temp1']*fqsm['nincr_temp2'])
                      + (fqsm['nincr_temp1']*fqsm['nincr_temp2']*fqsm['nincr_temp3'])
                      + (fqsm['nincr_temp1']*fqsm['nincr_temp2']*fqsm['nincr_temp3']*fqsm['nincr_temp4'])
                      + (fqsm['nincr_temp1']*fqsm['nincr_temp2']*fqsm['nincr_temp3']*fqsm['nincr_temp4']*fqsm['nincr_temp5'])
                      + (fqsm['nincr_temp1']*fqsm['nincr_temp2']*fqsm['nincr_temp3']*fqsm['nincr_temp4']*fqsm['nincr_temp5']*fqsm['nincr_temp6'])
                      + (fqsm['nincr_temp1']*fqsm['nincr_temp2']*fqsm['nincr_temp3']*fqsm['nincr_temp4']*fqsm['nincr_temp5']*fqsm['nincr_temp6']*fqsm['nincr_temp7'])
                      + (fqsm['nincr_temp1']*fqsm['nincr_temp2']*fqsm['nincr_temp3']*fqsm['nincr_temp4']*fqsm['nincr_temp5']*fqsm['nincr_temp6']*fqsm['nincr_temp7']*fqsm['nincr_temp8']))

fqsm = fqsm.drop(['ibq_l1', 'ibq_l2', 'ibq_l3', 'ibq_l4', 'ibq_l5', 'ibq_l6', 'ibq_l7', 'ibq_l8', 'nincr_temp1',
                            'nincr_temp2', 'nincr_temp3', 'nincr_temp4', 'nincr_temp5', 'nincr_temp6', 'nincr_temp7',
                            'nincr_temp8'], axis=1)



#noa
fqsm['atq_l4'] = fqsm.groupby(['gvkey'])['atq'].shift(4)
fqsm['ivaoq'] = np.where(fqsm['ivaoq'].isnull(), 0, 1)
fqsm['dlcq'] = np.where(fqsm['dlcq'].isnull(), 0, 1)
fqsm['dlttq'] = np.where(fqsm['dlttq'].isnull(), 0, 1)
fqsm['mibq'] = np.where(fqsm['mibq'].isnull(), 0, 1)
fqsm['pstkq'] = np.where(fqsm['pstkq'].isnull(), 0, 1)
fqsm['noa'] = (fqsm['atq']-fqsm['cheq']-fqsm['ivaoq'])-                 (fqsm['atq']-fqsm['dlcq']-fqsm['dlttq']-fqsm['mibq']-fqsm['pstkq']-fqsm['ceqq'])/fqsm['atq_l4']



# op
fqsm['xintq0'] = np.where(fqsm['xintq'].isnull(), 0, fqsm['xintq'])
fqsm['xsgaq0'] = np.where(fqsm['xsgaq'].isnull(), 0, fqsm['xsgaq'])
fqsm['beq'] = np.where(fqsm['seqq']>0, fqsm['seqq']+0-fqsm['pstkq'], np.nan)
fqsm['beq'] = np.where(fqsm['beq']<=0, np.nan, fqsm['beq'])
fqsm['beq_l4'] = fqsm.groupby(['gvkey'])['beq'].shift(4)
fqsm['op'] = (ttm4('revtq', fqsm)-ttm4('cogsq', fqsm)-ttm4('xsgaq0', fqsm)-ttm4('xintq0', fqsm))/fqsm['beq_l4']


#sgr
fqsm['saleq4'] = ttm4('saleq', fqsm)
fqsm['saleq4'] = np.where(fqsm['saleq4'].isnull(), fqsm['saley'], fqsm['saleq4'])
fqsm['saleq4_l4'] = fqsm.groupby(['gvkey'])['saleq4'].shift(4)
fqsm['sgr'] = (fqsm['saleq4']/fqsm['saleq4_l4'])-1




#ni
# fqsm['sps'] = fqsm['cshoc'] * fqsm['ajexdi']
# fqsm['sps_l1'] = fqsm.groupby('gvkey')['sps'].shift(1)
# fqsm['ni'] = np.log(fqsm['sps']/fqsm['sps_l1'])




#rna
fqsm['noa_l4'] = fqsm.groupby(['gvkey'])['noa'].shift(4)
fqsm['rna'] = fqsm['oiadpq']/fqsm['noa_l4']





#roa
fqsm['atq_l1'] = fqsm.groupby(['gvkey'])['atq'].shift(1)
fqsm['roa'] = fqsm['ibq']/fqsm['atq_l1']





#roe
fqsm['ceqq_l1'] = fqsm.groupby(['gvkey'])['ceqq'].shift(1)
fqsm['roe'] = fqsm['ibq']/fqsm['ceqq_l1']



#rsup
fqsm['saleq_l4'] = fqsm.groupby(['gvkey'])['saleq'].shift(4)
fqsm['rsup'] = (fqsm['saleq'] - fqsm['saleq_l4'])/fqsm['me']




#seas1a
fqsm['seas1a'] = fqsm.groupby(['gvkey'])['retm'].shift(11)




#sp
fqsm['sp'] = fqsm['saleq4']/fqsm['me']



#acc
fqsm['acc'] = (fqsm['iby']-fqsm['oancfy'])/ttm4('atq',fqsm)


# dy
# fqsm['me_l1'] = fqsm.groupby(['gvkey'])['me'].shift(1)
# fqsm['retdy'] = fqsm['retm'] - fqsm['retx']
# fqsm['mdivpay'] = fqsm['retdy']*fqsm['me_l1']


#pctacc
fqsm['iby1'] = fqsm['iby'].replace(0,0.01)
fqsm['pctacc'] = (fqsm['iby']-fqsm['oancfy'])/abs(fqsm['iby1'])




#pm
fqsm['pm'] = ttm4('oiadpq',fqsm)/ttm4('saleq',fqsm)



print('fqsm')


from tqdm import tqdm




fqsm['date'] = fqsm.groupby(['gvkey'])['jdate'].shift(-1)

fqsm = pd.merge(secm[['retm','gvkey','jdate']], fqsm, how='left', on=['gvkey','jdate']) # secm can be changed to crsp_mom
fqsm['datadate_secm'] = fqsm.groupby(['gvkey'])['datadate_secm'].fillna(method='ffill')
fqsm[['gvkey1', 'datadate1']] = fqsm[['gvkey', 'datadate_secm']]  # avoid the bug of 'groupby' for py 3.8
fqsm = fqsm.groupby(['gvkey1','datadate1'], as_index=False).fillna(method='ffill')


fqsm = fqsm.rename(columns={"retm_x": "retm"})
###momentum####
def mom(start, end, df):
    """
    :param start: Order of starting lag
    :param end: Order of ending lag
    :param df: Dataframe
    :return: Momentum factor
    """
    lag = pd.DataFrame()
    result = 1
    for i in range(start, end):
        lag['mom%s' % i] = df.groupby(['gvkey'])['retm'].shift(i)
        result = result * (1+lag['mom%s' % i])
    result = result - 1
    return result



fqsm['mom12m'] = mom(1,12,fqsm)
fqsm['mom36m'] = mom(1,36,fqsm)
fqsm['mom60m'] = mom(12,60,fqsm)
fqsm['mom6m'] = mom(1,6,fqsm)
fqsm['mom1m'] = fqsm.groupby(['gvkey'])['retm'].shift(1)



def standardize(df):
    # exclude the the information columns
    col_names = df.columns.values.tolist()
    list_to_remove = ['gvkey', 'exchg', 'loc', 'fic', 'iid',
       'datadate_secm','retm', 'jdate', 'indfmt',
       'consol', 'popsrc', 'datafmt', 'datadate_fundq', 'pdateq',
       'fdateq', 'permno', 'jdate', 'date', 'datadate', 'sic', 'count', 'exchcd', 'shrcd', 'ffi49', 'ret',
       'retadj', 'retx', 'lag_me']
    col_names = list(set(col_names).difference(set(list_to_remove)))
    for col_name in tqdm(col_names):
        print('processing %s' % col_name)
        # count the non-missing number of factors, we only count non-missing values
        unique_count = df.dropna(subset=['%s' % col_name]).groupby(['date'])['%s' % col_name].unique().apply(len)
        unique_count = pd.DataFrame(unique_count).reset_index()
        unique_count.columns = ['date', 'count']
        df = pd.merge(df, unique_count, how='left', on=['date'])
        # ranking, and then standardize the data
        df['%s_rank' % col_name] = df.groupby(['date'])['%s' % col_name].rank(method='dense')
        df['rank_%s' % col_name] = (df['%s_rank' % col_name] - 1) / (df['count'] - 1) * 2 - 1
        df = df.drop(['%s_rank' % col_name, '%s' % col_name, 'count'], axis=1)
   # df = df.dropna()
    return df




df_rank = fqsm.copy()
df_rank['lag_me'] = df_rank['me']
df_rank = standardize(df_rank)




# df_rank.columns.values




# breakdown = df_rank.groupby(['jdate'])['rank_me'].describe(percentiles=[0.2,0.4,0.6,0.8]).reset_index()
# breakdown = breakdown[['jdate','20%','40%','60%','80%']]



# chars = pd.merge(df_rank, breakdown, how='left', on=['jdate'])

with open('chars_q_hkg.feather', 'wb') as f:
    feather.write_feather(fqsm, f)

# with open('chars_q_rank_hkg.feather', 'wb') as f:
#     feather.write_feather(df_rank[['gvkey', 'exchg', 'loc', 'retm', 'jdate','date', 'lag_me', 'rank_cp', 'rank_loq',
#                                'rank_intanq', 'rank_xsgaq0', 'rank_aoq', 'rank_mibq', 'rank_txdbq', 'rank_ep', 'rank_iby1',
#                                 'rank_mb', 'rank_xintq0', 'rank_op', 'rank_ppentq_l4', 'rank_saleq4', 'rank_me', 'rank_actq',
#                                 'rank_ato', 'rank_ltq', 'rank_saleq_l1', 'rank_prccd', 'rank_mom60m', 'rank_saleq4_l4',
#                                 'rank_cogsq', 'rank_xintq', 'rank_gdwlq', 'rank_invtq_l4', 'rank_ppentq_l3', 'rank_iby',
#                                 'rank_xsgaq', 'rank_sp', 'rank_sgr', 'rank_revtq4', 'rank_seqq', 'rank_alm', 'rank_cinvest',
#                                 'rank_txtq', 'rank_cf', 'rank_saleq', 'rank_ivaoq', 'rank_rna', 'rank_rectq_l4',
#                                 'rank_saleq_l2', 'rank_cash', 'rank_apq_l4', 'rank_pe', 'rank_pstkq', 'rank_revtq',
#                                 'rank_dpy', 'rank_ltq_l4', 'rank_roe', 'rank_apq', 'rank_rectq', 'rank_mom12m',
#                                 'rank_lcoq_l4', 'rank_lev', 'rank_depr', 'rank_cshoc', 'rank_ppentq_l1', 
#                                 'rank_loq_l4', 'rank_cogsq4', 'rank_oancfy', 'rank_saleq_l3', 'rank_dpq', 
#                                 'rank_ceqq_l1', 'rank_chpm', 'rank_ppentq_l2', 'rank_noa_l4', 'rank_agr', 'rank_acoq', 
#                                 'rank_mom6m', 'rank_seas1a', 'rank_saleq_l4', 'rank_dlttq', 'rank_atq_l4', 'rank_ceqq', 
#                                 'rank_rsup', 'rank_invtq', 'rank_nincr', 'rank_mom36m', 'rank_ibq', 'rank_ibq4', 'rank_ala', 
#                                 'rank_lcoq', 'rank_cheq', 'rank_cashdebt', 'rank_txtq_l4', 'rank_beq_l4', 'rank_ppentq', 'rank_atq', 
#                                 'rank_mom1m', 'rank_saleq4_l1', 'rank_pm', 'rank_acoq_l4', 'rank_beq', 'rank_oiadpq', 'rank_chtx', 
#                                 'rank_dlcq', 'rank_atq_l1', 'rank_pctacc', 'rank_fyr', 'rank_acc', 'rank_lgr', 'rank_noa', 'rank_grltnoa', 
#                                 'rank_roa', 'rank_saley', 'rank_ibq4_l1', 'rank_earnings', 'rank_bm', 'rank_gma', 'rank_me']], f)

with open('chars_q_rank_hkg.feather', 'wb') as f:
    feather.write_feather(df_rank,f)