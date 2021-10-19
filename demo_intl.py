
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



# fundq = pd.read_pickle('./fundq_hkg_1990.pkl')

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
fqsm['ala'] = fqsm['cheq'] + 0.75*(fqsm['actq']-fqsm['cheq'])+                 0.5*(fqsm['atq']-fqsm['actq']-fqsm['gdwlq']-fqsm['intanq'])
fqsm['alm'] = fqsm['ala']/(fqsm['atq']+fqsm['me']-fqsm['ceqq'])




# ato
fqsm['noa'] = (fqsm['atq']-fqsm['cheq']-fqsm['ivaoq'])-                 (fqsm['atq']-fqsm['dlcq']-fqsm['dlttq']-fqsm['mibq']-fqsm['pstkq']-fqsm['ceqq'])/fqsm['atq_l4']
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

fqsm['cinvest'] = ((fqsm['ppentq'] - fqsm['ppentq_l1']) / fqsm['saleq'])                       -(fqsm[['c_temp1', 'c_temp2', 'c_temp3']].mean(axis=1))
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

'''
#########anual chars#########

# funda = conn.raw_sql("""
#                      select *
#                     from comp.g_funda
#                      where datadate > '01/01/1990'
#                      and loc = 'HKG'
#                      """)

# funda = funda.sort_values(['gvkey','datadate','iid','isin','sedol']) # order by gvkey, date, issue id, other id's
# with open('./funda_hkg_1990.pkl', 'wb') as f:
#  pkl.dump(funda, f)

# funda = pd.read_pickle('./funda_hkg_1990.pkl')


# with open('funda_hkg.feather', 'rb') as f:
#   funda = feather.read_feather(f)




# funda = funda[~funda['isin'].isna()]   # international id
# funda = funda[~funda['sedol'].isna()]   # international id
# funda = funda[ (funda['exchg'] == 249) | (funda['exchg'] == 250)] # shanghai / shenzhen





funda = funda[funda['datadate'] >= datetime.datetime.strptime('1990-01-01','%Y-%m-%d').date()]





funda = funda.sort_values(['gvkey','datadate','exchg','isin','sedol','seq'])





# In[221]:


# print(funda.shape)
# # funda = funda[~funda['seq'].isna()]
# # funda = funda[~funda['ib'].isna()]
# print(funda.shape)




funda['txdb'] = funda['txdb'].fillna(0)
funda['txt'] = funda['txt'].fillna(0)
funda['pstk'] = funda['pstk'].fillna(0)

funda['mib'] = funda['mib'].fillna(0)
funda['dlc'] = funda['dlc'].fillna(0)
funda['ivao'] = funda['ivao'].fillna(0)
funda['dltt'] = funda['dltt'].fillna(0)



#be

funda['be'] = funda['seq'] + funda['txdb'] + funda['txt'] - funda['pstk']







# # Merge funda and secm



# funda['datadate'] = pd.to_datetime(funda['datadate'])
# # join date is jdate
# # quarterly fundamentals are expected to report later than the datadate
# # 3 month is enough for the reporting process
# # thus, we don't have forseeing-data problem
# funda['jdate'] = funda['datadate'] + MonthEnd(3)
# funda = funda.sort_values(['gvkey','datadate','exchg','isin','sedol'])




fundamental_varlist=[
    # id
    'gvkey', 'indfmt', 'consol', 'popsrc', 'datafmt','exchg', 'loc','fic', 'sedol', 'isin','datadate','pdate','fdate','fyr', 'sich',
    # varaibles we want 
    'ib',
    'seq','txdb','txt','pstk','dp','at',
    'che','act','gdwl','intan','ceq',
    'ivao','dlc','dltt','mib','sale',
    'lt','ppent','revt','cogs',
    'rect','aco','ap','lco','lo','invt','ao','xint','xsga','be', 'oiadp', 'oancf', 'lct', 'np', 'txp'
    ]
funda = funda[fundamental_varlist]





funda['datadate'] = pd.to_datetime(funda['datadate'])
# join date is jdate
# quarterly fundamentals are expected to report later than the datadate
# 3 month is enough for the reporting process
# thus, we don't have forseeing-data problem
funda['jdate'] = funda['datadate'] + MonthEnd(6)
funda = funda.sort_values(['gvkey','datadate','exchg','isin','sedol'])







fasm = pd.merge(secm, funda, how='left', on=['gvkey','jdate','exchg','isin','sedol','loc','fic'])

print('merge2')





# # Forward Fill the Fundq info to Empty Month




# fasm.columns.values




fasm.columns = ['gvkey', 'exchg', 'loc', 'fic', 'iid', 'sedol', 'isin',
       'datadate_secm', 'cshoc', 'prccd', 'me', 'retm', 'jdate', 'indfmt',
       'consol', 'popsrc', 'datafmt', 'datadate_funda', 'pdate', 'fdate',
       'fyr', 'sich', 'ib', 'seq', 'txdb', 'txt', 'pstk', 'dp', 'at', 'che',
       'act', 'gdwl', 'intan', 'ceq', 'ivao', 'dlc', 'dltt', 'mib',
       'sale', 'lt', 'ppent', 'revt', 'cogs', 'rect', 'aco', 'ap', 'lco',
       'lo', 'invt', 'ao', 'xint', 'xsga','be', 'oiadp', 'oancf', 'lct', 'np', 'txp']




fasm = fasm.sort_values(['gvkey','jdate','isin','sedol']) # order by gvkey, date, issue id, other id's





fasm['pdate'] = fasm.groupby('gvkey')['pdate'].fillna(method='ffill')
fasm['fdate'] = fasm.groupby('gvkey')['fdate'].fillna(method='ffill')
fasm['ib'] = fasm.groupby('gvkey')['ib'].fillna(method='ffill')
fasm['be'] = fasm.groupby('gvkey')['be'].fillna(method='ffill')




fasm['me'] = fasm['me']/1e6 # 1e6 is one million





fasm['bm'] = fasm['be']/fasm['me']
fasm['mb'] = fasm['me']/fasm['be']

#ffi49
fasm = fasm.rename(columns={'sich':'sic'})
fasm['ffi49'] = ffi49(fasm)
fasm['ffi49'] = fasm['ffi49'].fillna(49)
fasm['ffi49'] = fasm['ffi49'].astype(int)

# bm_ia
df = fasm.dropna(subset =['datadate_funda'])
df_temp = df.groupby(['datadate_funda', 'ffi49'])['bm'].mean().reset_index()
df_temp = df_temp.rename(columns={'bm': 'bm_ind'})
fasm = pd.merge(fasm, df_temp, how='left', on=['datadate_funda', 'ffi49'])
fasm['bm_ia'] = fasm['bm']/fasm['bm_ind']

# me_ia
df = fasm.dropna(subset =['datadate_funda'])
df_temp = df.groupby(['datadate_funda', 'ffi49'])['me'].mean().reset_index()
df_temp = df_temp.rename(columns={'me': 'me_ind'})
fasm = pd.merge(fasm, df_temp, how='left', on=['datadate_funda', 'ffi49'])
fasm['me_ia'] = fasm['me']/fasm['me_ind']


# ep
fasm['ep'] = fasm['ib']/fasm['me']


#dp fillna
# cp
fasm['dp'] = fasm.groupby('gvkey')['dp'].fillna(method='ffill')
fasm['cf'] = fasm['ib'] + fasm['dp']
fasm['cp'] = fasm['cf']/fasm['me']


# agr
fasm['at'] = fasm.groupby('gvkey')['at'].fillna(method='ffill')
fasm['at_l1'] = fasm.groupby('gvkey')['at'].shift(12)
fasm['agr'] = (fasm['at'] - fasm['at_l1']) / fasm['at_l1']



# alm
fasm['che'] = fasm.groupby('gvkey')['che'].fillna(method='ffill')
fasm['act'] = fasm.groupby('gvkey')['act'].fillna(method='ffill')
fasm['gdwl'] = fasm.groupby('gvkey')['gdwl'].fillna(method='ffill')
fasm['intan'] = fasm.groupby('gvkey')['intan'].fillna(method='ffill')
fasm['ceq'] = fasm.groupby('gvkey')['ceq'].fillna(method='ffill')
fasm['ala'] = fasm['che'] + 0.75*(fasm['act']-fasm['che'])+0.5*(fasm['at']-fasm['act']-fasm['gdwl']-fasm['intan'])
fasm['alm'] = fasm['ala']/(fasm['at']+fasm['me']-fasm['ceq'])




# ato
fasm['ivao'] = fasm.groupby('gvkey')['ivao'].fillna(method='ffill')
fasm['dlc'] = fasm.groupby('gvkey')['dlc'].fillna(method='ffill')
fasm['dltt'] = fasm.groupby('gvkey')['dltt'].fillna(method='ffill')
fasm['mib'] = fasm.groupby('gvkey')['mib'].fillna(method='ffill')
fasm['pstk'] = fasm.groupby('gvkey')['pstk'].fillna(method='ffill')
fasm['sale'] = fasm.groupby('gvkey')['sale'].fillna(method='ffill')
fasm['noa'] = (fasm['at']-fasm['che']-fasm['ivao'])-(fasm['at']-fasm['dlc']-fasm['dltt']-fasm['mib']-fasm['pstk']-fasm['ceq'])/fasm['at_l1']
fasm['noa_l1'] = fasm.groupby(['gvkey'])['noa'].shift(12)
fasm['ato'] = fasm['sale']/fasm['noa_l1']




# cash
fasm['cash'] = fasm['che']/fasm['at']





# cashdebt
fasm['lt'] = fasm.groupby(['gvkey'])['lt'].fillna(method='ffill')
fasm['lt_l1'] = fasm.groupby(['gvkey'])['lt'].shift(12)
fasm['cashdebt'] = (fasm['ib'] + fasm['dp'])/((fasm['lt']+fasm['lt_l1'])/2)



#chpm
fasm['ib_l1'] = fasm.groupby(['gvkey'])['ib'].shift(12)
fasm['sale_l1'] = fasm.groupby(['gvkey'])['sale'].shift(12)
fasm['chpm'] = (fasm['ib']/fasm['sale'])-(fasm['ib_l1']/fasm['sale_l1'])



#chtx
fasm['txt'] = fasm.groupby('gvkey')['txt'].fillna(method='ffill')
fasm['txt_l1'] = fasm.groupby(['gvkey'])['txt'].shift(12)
fasm['at_l1'] = fasm.groupby(['gvkey'])['at'].shift(12)
fasm['chtx'] = (fasm['txt']-fasm['txt_l1'])/fasm['at_l1']



# #cinvest
# fasm['ppent'] = fasm.groupby('gvkey')['ppent'].fillna(method='ffill')
# fasm['ppent_l1'] = fasm.groupby(['gvkey'])['ppent'].shift(3)
# fasm['ppent_l2'] = fasm.groupby(['gvkey'])['ppent'].shift(6)
# fasm['ppent_l3'] = fasm.groupby(['gvkey'])['ppent'].shift(9)
# fasm['ppent_l4'] = fasm.groupby(['gvkey'])['ppent'].shift(12)
# fasm['sale_l1'] = fasm.groupby(['gvkey'])['sale'].shift(3)
# fasm['sale_l2'] = fasm.groupby(['gvkey'])['sale'].shift(6)
# fasm['sale_l3'] = fasm.groupby(['gvkey'])['sale'].shift(9)

# fasm['c_temp1'] = (fasm['ppent_l1'] - fasm['ppent_l2']) / fasm['sale_l1']
# fasm['c_temp2'] = (fasm['ppent_l2'] - fasm['ppent_l3']) / fasm['sale_l2']
# fasm['c_temp3'] = (fasm['ppent_l3'] - fasm['ppent_l4']) / fasm['sale_l3']

# fasm['c_temp1'] = (fasm['ppent_l1'] - fasm['ppent_l2']) / 0.01
# fasm['c_temp2'] = (fasm['ppent_l2'] - fasm['ppent_l3']) / 0.01
# fasm['c_temp3'] = (fasm['ppent_l3'] - fasm['ppent_l4']) / 0.01

# fasm['cinvest'] = ((fasm['ppent'] - fasm['ppent_l1']) / fasm['sale'])                       -(fasm[['c_temp1', 'c_temp2', 'c_temp3']].mean(axis=1))
# fasm['cinvest'] = np.where(fasm['sale']<=0, ((fasm['ppent'] - fasm['ppent_l1']) / 0.01)
#                                 -(fasm[['c_temp1', 'c_temp2', 'c_temp3']].mean(axis=1)), fasm['cinvest'])

# fasm = fasm.drop(['c_temp1', 'c_temp2', 'c_temp3'], axis=1)



#depr
fasm['depr'] = fasm['dp']/fasm['ppent']





#gma
fasm['revt'] = fasm.groupby('gvkey')['revt'].fillna(method='ffill')
fasm['cogs'] = fasm.groupby('gvkey')['cogs'].fillna(method='ffill')
fasm['at_l1'] = fasm.groupby(['gvkey'])['at'].shift(12)
fasm['gma'] = (fasm['revt']-fasm['cogs'])/fasm['at_l1']



#grltnoa
fasm['rect'] = fasm.groupby('gvkey')['rect'].fillna(method='ffill')
fasm['aco'] = fasm.groupby('gvkey')['aco'].fillna(method='ffill')
fasm['ap'] = fasm.groupby('gvkey')['ap'].fillna(method='ffill')
fasm['lco'] = fasm.groupby('gvkey')['lco'].fillna(method='ffill')
fasm['lo'] = fasm.groupby('gvkey')['lo'].fillna(method='ffill')
fasm['invt'] = fasm.groupby('gvkey')['invt'].fillna(method='ffill')
fasm['ao'] = fasm.groupby('gvkey')['ao'].fillna(method='ffill')

fasm['rect_l1'] = fasm.groupby(['gvkey'])['rect'].shift(12)
fasm['aco_l1'] = fasm.groupby(['gvkey'])['aco'].shift(12)
fasm['ap_l1'] = fasm.groupby(['gvkey'])['ap'].shift(12)
fasm['lco_l1'] = fasm.groupby(['gvkey'])['lco'].shift(12)
fasm['lo_l1'] = fasm.groupby(['gvkey'])['lo'].shift(12)
fasm['invt_l1'] = fasm.groupby(['gvkey'])['invt'].shift(12)
fasm['ppent_l1'] = fasm.groupby(['gvkey'])['ppent'].shift(12)
fasm['at_l1'] = fasm.groupby(['gvkey'])['at'].shift(12)

fasm['grltnoa'] = ((fasm['rect']+fasm['invt']+fasm['ppent']+fasm['aco']+fasm['intan']+
                       fasm['ao']-fasm['ap']-fasm['lco']-fasm['lo'])-
                      (fasm['rect_l1']+fasm['invt_l1']+fasm['ppent_l1']+fasm['aco_l1']-fasm['ap_l1']-fasm['lco_l1']-fasm['lo_l1'])-\
                     (fasm['rect']-fasm['rect_l1']+fasm['invt']-fasm['invt_l1']+fasm['aco']-
                      (fasm['ap']-fasm['ap_l1']+fasm['lco']-fasm['lco_l1'])-
                      fasm['dp']))/((fasm['at']+fasm['at_l1'])/2)



#lev
fasm['lev'] = fasm['lt']/fasm['me']

#lgr
fasm['lt_l1'] = fasm.groupby(['gvkey'])['lt'].shift(12)
fasm['lgr'] = (fasm['lt']/fasm['lt_l1'])-1



# #nincr
# fasm['ib_l1'] = fasm.groupby(['gvkey'])['ib'].shift(3)
# fasm['ib_l2'] = fasm.groupby(['gvkey'])['ib'].shift(6)
# fasm['ib_l3'] = fasm.groupby(['gvkey'])['ib'].shift(9)
# fasm['ib_l4'] = fasm.groupby(['gvkey'])['ib'].shift(12)
# fasm['ib_l5'] = fasm.groupby(['gvkey'])['ib'].shift(15)
# fasm['ib_l6'] = fasm.groupby(['gvkey'])['ib'].shift(18)
# fasm['ib_l7'] = fasm.groupby(['gvkey'])['ib'].shift(21)
# fasm['ib_l8'] = fasm.groupby(['gvkey'])['ib'].shift(24)

# fasm['nincr_temp1'] = np.where(fasm['ib'] > fasm['ib_l1'], 1, 0)
# fasm['nincr_temp2'] = np.where(fasm['ib_l1'] > fasm['ib_l2'], 1, 0)
# fasm['nincr_temp3'] = np.where(fasm['ib_l2'] > fasm['ib_l3'], 1, 0)
# fasm['nincr_temp4'] = np.where(fasm['ib_l3'] > fasm['ib_l4'], 1, 0)
# fasm['nincr_temp5'] = np.where(fasm['ib_l4'] > fasm['ib_l5'], 1, 0)
# fasm['nincr_temp6'] = np.where(fasm['ib_l5'] > fasm['ib_l6'], 1, 0)
# fasm['nincr_temp7'] = np.where(fasm['ib_l6'] > fasm['ib_l7'], 1, 0)
# fasm['nincr_temp8'] = np.where(fasm['ib_l7'] > fasm['ib_l8'], 1, 0)

# fasm['nincr'] = (fasm['nincr_temp1']
#                       + (fasm['nincr_temp1']*fasm['nincr_temp2'])
#                       + (fasm['nincr_temp1']*fasm['nincr_temp2']*fasm['nincr_temp3'])
#                       + (fasm['nincr_temp1']*fasm['nincr_temp2']*fasm['nincr_temp3']*fasm['nincr_temp4'])
#                       + (fasm['nincr_temp1']*fasm['nincr_temp2']*fasm['nincr_temp3']*fasm['nincr_temp4']*fasm['nincr_temp5'])
#                       + (fasm['nincr_temp1']*fasm['nincr_temp2']*fasm['nincr_temp3']*fasm['nincr_temp4']*fasm['nincr_temp5']*fasm['nincr_temp6'])
#                       + (fasm['nincr_temp1']*fasm['nincr_temp2']*fasm['nincr_temp3']*fasm['nincr_temp4']*fasm['nincr_temp5']*fasm['nincr_temp6']*fasm['nincr_temp7'])
#                       + (fasm['nincr_temp1']*fasm['nincr_temp2']*fasm['nincr_temp3']*fasm['nincr_temp4']*fasm['nincr_temp5']*fasm['nincr_temp6']*fasm['nincr_temp7']*fasm['nincr_temp8']))

# fasm = fasm.drop(['ib_l1', 'ib_l2', 'ib_l3', 'ib_l4', 'ib_l5', 'ib_l6', 'ib_l7', 'ib_l8', 'nincr_temp1',
#                             'nincr_temp2', 'nincr_temp3', 'nincr_temp4', 'nincr_temp5', 'nincr_temp6', 'nincr_temp7',
#                             'nincr_temp8'], axis=1)



#noa
fasm['at_l1'] = fasm.groupby(['gvkey'])['at'].shift(12)
fasm['ivao'] = np.where(fasm['ivao'].isnull(), 0, 1)
fasm['dlc'] = np.where(fasm['dlc'].isnull(), 0, 1)
fasm['dltt'] = np.where(fasm['dltt'].isnull(), 0, 1)
fasm['mib'] = np.where(fasm['mib'].isnull(), 0, 1)
fasm['pstk'] = np.where(fasm['pstk'].isnull(), 0, 1)
fasm['noa'] = (fasm['at']-fasm['che']-fasm['ivao'])-(fasm['at']-fasm['dlc']-fasm['dltt']-fasm['mib']-fasm['pstk']-fasm['ceq'])/fasm['at_l1']



# op
fasm['xint'] = fasm.groupby('gvkey')['xint'].fillna(method='ffill')
fasm['xsga'] = fasm.groupby('gvkey')['xsga'].fillna(method='ffill')
fasm['seq'] = fasm.groupby('gvkey')['seq'].fillna(method='ffill')
fasm['xint0'] = np.where(fasm['xint'].isnull(), 0, fasm['xint'])
fasm['xsga0'] = np.where(fasm['xsga'].isnull(), 0, fasm['xsga'])
fasm['cogs0'] = np.where(fasm['cogs'].isnull(), 0, fasm['cogs'])
# fasm['be'] = np.where(fasm['seq']>0, fasm['seq']+0-fasm['pstk'], np.nan)
# fasm['be'] = np.where(fasm['be']<=0, np.nan, fasm['be'])
# fasm['be_l1'] = fasm.groupby(['gvkey'])['be'].shift(12)
# fasm['op'] = (ttm4('revt', fasm)-ttm4('cogs', fasm)-ttm4('xsga0', fasm)-ttm4('xint0', fasm))/fasm['be_l1']
condlist = [fasm['revt'].isnull(), fasm['be'].isnull()]
choicelist = [np.nan, np.nan]
fasm['op'] = np.select(condlist, choicelist,
                          default=(fasm['revt'] - fasm['cogs0'] - fasm['xsga0'] - fasm['xint0'])/fasm['be'])


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




fasm['mom12m'] = mom(1,12,fasm)
fasm['mom36m'] = mom(1,36,fasm)
fasm['mom60m'] = mom(12,60,fasm)
fasm['mom6m'] = mom(1,6,fasm)
fasm['mom1m'] = fasm['retm']




#sgr
fasm['sale_l1'] = fasm.groupby(['gvkey'])['sale'].shift(12)
fasm['sgr'] = (fasm['sale']/fasm['sale_l1'])-1




#ni
# fasm['sps'] = fasm['cshoc'] * fasm['ajexdi']
# fasm['sps_l1'] = fasm.groupby('gvkey')['sps'].shift(3)
# fasm['ni'] = np.log(fasm['sps']/fasm['sps_l1'])




#rna
fasm['oiadp'] = fasm.groupby('gvkey')['oiadp'].fillna(method='ffill')
fasm['noa_l1'] = fasm.groupby(['gvkey'])['noa'].shift(12)
fasm['rna'] = fasm['oiadp']/fasm['noa_l1']





#roa
fasm['at_l1'] = fasm.groupby(['gvkey'])['at'].shift(12)
fasm['roa'] = fasm['ib']/fasm['at_l1']





#roe
fasm['ceq_l1'] = fasm.groupby(['gvkey'])['ceq'].shift(12)
fasm['roe'] = fasm['ib']/fasm['ceq_l1']



#rsup
fasm['sale_l1'] = fasm.groupby(['gvkey'])['sale'].shift(12)
fasm['rsup'] = (fasm['sale'] - fasm['sale_l1'])/fasm['me']




#seas1a
fasm['seas1a'] = fasm.groupby(['gvkey'])['retm'].shift(11)




#sp
fasm['sp'] = fasm['sale']/fasm['me']



#acc
# fasm['ib'] = fasm.groupby('gvkey')['ib'].fillna(method='ffill')
# fasm['oancf'] = fasm.groupby('gvkey')['oancf'].fillna(method='ffill')
# fasm['acc'] = (fasm['ib']-fasm['oancf'])/fasm['at']
fasm['lct'] = fasm.groupby('gvkey')['lct'].fillna(method='ffill')
fasm['np'] = fasm.groupby('gvkey')['np'].fillna(method='ffill')
fasm['act_l1'] = fasm.groupby(['gvkey'])['act'].shift(12)
fasm['lct_l1'] = fasm.groupby(['gvkey'])['lct'].shift(12)
condlist = [fasm['np'].isnull(),
          fasm['act'].isnull() | fasm['lct'].isnull()]
choicelist = [((fasm['act']-fasm['lct'])-(fasm['act_l1']-fasm['lct_l1'])/(10*fasm['be'])),
              (fasm['ib']-fasm['oancf'])/(10*fasm['be'])]
fasm['acc'] = np.select(condlist,
                            choicelist,
                            default=((fasm['act']-fasm['lct']+fasm['np'])-
                                     (fasm['act_l1']-fasm['lct_l1']+fasm['np'].shift(12)))/(10*fasm['be']))

# dy
# fasm['me_l1'] = fasm.groupby(['gvkey'])['me'].shift(3)
# fasm['retdy'] = fasm['retm'] - fasm['retx']
# fasm['mdivpay'] = fasm['retdy']*fasm['me_l1']


#pctacc
# fasm['iby1'] = fasm['iby'].replace(0,0.01)
# fasm['pctacc'] = (fasm['iby']-fasm['oancfy'])/abs(fasm['iby1'])
fasm['txp'] = fasm.groupby('gvkey')['txp'].fillna(method='ffill')
fasm['che_l1'] = fasm.groupby(['gvkey'])['che'].shift(12)
fasm['dlc_l1'] = fasm.groupby(['gvkey'])['dlc'].shift(12)
fasm['txp_l1'] = fasm.groupby(['gvkey'])['txp'].shift(12)

condlist = [fasm['ib']==0,
            fasm['oancf'].isnull(),
            fasm['oancf'].isnull() & fasm['ib']==0]
choicelist = [(fasm['ib']-fasm['oancf'])/0.01,
              ((fasm['act'] - fasm['act_l1']) - (fasm['che'] - fasm['che_l1']))-
              ((fasm['lct'] - fasm['lct_l1']) - (fasm['dlc']) - fasm['dlc_l1']-
               ((fasm['txp'] - fasm['txp_l1']) - fasm['dp']))/fasm['ib'].abs(),
              ((fasm['act'] - fasm['act_l1']) - (fasm['che'] - fasm['che_l1'])) -
              ((fasm['lct'] - fasm['lct_l1']) - (fasm['dlc']) - fasm['dlc_l1'] -
               ((fasm['txp'] - fasm['txp_l1']) - fasm['dp']))]
fasm['pctacc'] = np.select(condlist, choicelist, default=(fasm['ib']-fasm['oancf'])/fasm['ib'].abs())




#pm
fasm['pm'] = fasm['oiadp']/fasm['sale']

print('fasm')

def standardize1(df):
    # exclude the the information columns
    col_names = df.columns.values.tolist()
    list_to_remove = ['gvkey', 'exchg', 'loc', 'fic', 'iid', 'sedol', 'isin',
       'datadate_secm','retm', 'jdate', 'indfmt',
       'consol', 'popsrc', 'datafmt', 'datadate_funda', 'pdate',
       'fdate', 'permno', 'jdate', 'date', 'datadate', 'sic', 'count', 'exchcd', 'shrcd', 'ffi49', 'ret',
       'retadj', 'retx', 'lag_me','']
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



fasm['date'] = fasm.groupby(['gvkey'])['jdate'].shift(-1)
df_ranka = fasm.copy()
df_ranka['lag_me'] = df_ranka['me']
df_ranka = standardize1(df_ranka)
charsa = df_ranka

with open('chars_a_sg1990.feather', 'wb') as f:
    feather.write_feather(fasm, f)

with open('chars_a_rank_sg1990.feather', 'wb') as f:
    feather.write_feather(df_ranka[['gvkey', 'exchg', 'loc', 'retm', 'jdate', 'date', 'lag_me',
                              'rank_intan', 'rank_cp', 'rank_invt_l1', 'rank_ivao', 'rank_rect',
                              'rank_cogs0', 'rank_act', 'rank_dlc_l1', 'rank_ep', 'rank_mb', 'rank_op',
                              'rank_noa_l1', 'rank_ap_l1', 'rank_me', 'rank_rect_l1', 'rank_ato',
                              'rank_cogs', 'rank_txt', 'rank_oancf', 'rank_prccd', 'rank_mom60m', 
                              'rank_dltt', 'rank_ap', 'rank_lo', 'rank_txp', 'rank_revt', 'rank_seq', 
                              'rank_lct_l1', 'rank_sp', 'rank_sgr', 'rank_txt_l1', 'rank_lo_l1', 'rank_alm', 
                              'rank_cf', 'rank_pstk', 'rank_rna', 'rank_txp_l1', 'rank_oiadp', 'rank_sale_l1', 
                              'rank_dp', 'rank_xint', 'rank_cash', 'rank_roe', 'rank_txdb', 'rank_xint0', 'rank_ceq',
                              'rank_mom12m', 'rank_lco', 'rank_lev', 'rank_mib', 'rank_xsga', 'rank_depr', 'rank_cshoc', 
                              'rank_dlc', 'rank_che', 'rank_aco_l1', 'rank_che_l1', 'rank_chpm', 'rank_agr', 'rank_mom6m', 
                              'rank_sale', 'rank_gdwl', 'rank_aco', 'rank_seas1a', 'rank_ppent_l1', 'rank_lco_l1',
                              'rank_ppent', 'rank_rsup', 'rank_xsga0', 'rank_mom36m', 'rank_ib', 'rank_lct', 'rank_ala',
                              'rank_be', 'rank_cashdebt', 'rank_act_l1', 'rank_ib_l1', 'rank_mom1m', 'rank_pm', 'rank_at',
                              'rank_chtx', 'rank_lt_l1', 'rank_ceq_l1', 'rank_invt', 'rank_pctacc', 'rank_fyr', 'rank_acc',
                              'rank_lgr', 'rank_noa', 'rank_grltnoa', 'rank_ao', 'rank_roa', 'rank_np', 'rank_lt', 'rank_at_l1',
                              'rank_bm', 'rank_gma','rank_bm_ia','rank_me_ia']], f)




# def wavg(group, avg_name, weight_name):
#     d = group[avg_name]
#     w = group[weight_name]
#     try:
#         return (d * w).sum() / w.sum()
#     except ZeroDivisionError:
#         return np.nan
    
# def plot3_a(col):
#     charsa['test'] = charsa['rank_%s'%col]
#     charsa['char_port'] = 1 + np.where(charsa['test']>-0.6,1,0) + np.where(charsa['test']>-0.2,1,0) + np.where(charsa['test']>0.2,1,0) + np.where(charsa['test']>0.6,1,0)
#     vwret = charsa.groupby(['jdate', 'char_port']).apply(wavg, 'retm', 'lag_me').to_frame().reset_index().rename(columns={0: 'vwret'})

#     vwmkt = charsa.groupby(['jdate']).apply(wavg, 'retm', 'lag_me').to_frame()
#     vwmkt = vwmkt.reset_index()
#     vwmkt['jdate'] = pd.to_datetime(vwmkt['jdate'])

#     # vwret['jdate'] = pd.to_datetime(vwret['jdate'])

#     vwret = vwret[vwret['jdate'].dt.year>=2001]

#     # figure 1 cumsum ret
#     plt.figure(figsize=(15,5), dpi=80)
#     plt.figure(1)
#     plt.clf()
#     plt.subplot(131)
#     l1 = plt.plot(vwret[vwret['char_port'] == 1]['jdate'], vwret[vwret['char_port'] == 1]['vwret'].cumsum(), label='port1')
#     l2 = plt.plot(vwret[vwret['char_port'] == 2]['jdate'], vwret[vwret['char_port'] == 2]['vwret'].cumsum(), label='port2')
#     l3 = plt.plot(vwret[vwret['char_port'] == 3]['jdate'], vwret[vwret['char_port'] == 3]['vwret'].cumsum(), label='port3')
#     l4 = plt.plot(vwret[vwret['char_port'] == 4]['jdate'], vwret[vwret['char_port'] == 4]['vwret'].cumsum(), label='port4')
#     l5 = plt.plot(vwret[vwret['char_port'] == 5]['jdate'], vwret[vwret['char_port'] == 5]['vwret'].cumsum(), label='port5')
#     mkt = plt.plot(vwmkt['jdate'], vwmkt[0].cumsum(), label='mkt')
#     plt.title('rank_%s_a'%col)
#     plt.legend()
#     # figure 2
#     # plt.figure(132)
#     plt.subplot(132)
#     plt.bar('port1',vwret[vwret['char_port']==1]['vwret'].mean(),label='port1')
#     plt.bar('port2',vwret[vwret['char_port']==2]['vwret'].mean(),label='port2')
#     plt.bar('port3',vwret[vwret['char_port']==3]['vwret'].mean(),label='port3')
#     plt.bar('port4',vwret[vwret['char_port']==4]['vwret'].mean(),label='port4')
#     plt.bar('port5',vwret[vwret['char_port']==5]['vwret'].mean(),label='port5')
#     plt.bar('mkt',vwret['vwret'].mean())
#     plt.title('%s_avg_ret_a'%col)
#     plt.legend()
#     # figure 3
#     # plt.figure(133)
#     plt.subplot(133)
#     plt.plot(charsa[(charsa['jdate'].dt.year>2000) & (charsa['jdate'].dt.year<=2020)][charsa['char_port']==1].groupby(['jdate'])['rank_%s'%col].count(),label='port1')
#     plt.plot(charsa[(charsa['jdate'].dt.year>2000) & (charsa['jdate'].dt.year<=2020)][charsa['char_port']==2].groupby(['jdate'])['rank_%s'%col].count(),label='port2')
#     plt.plot(charsa[(charsa['jdate'].dt.year>2000) & (charsa['jdate'].dt.year<=2020)][charsa['char_port']==3].groupby(['jdate'])['rank_%s'%col].count(),label='port3')
#     plt.plot(charsa[(charsa['jdate'].dt.year>2000) & (charsa['jdate'].dt.year<=2020)][charsa['char_port']==4].groupby(['jdate'])['rank_%s'%col].count(),label='port4')
#     plt.plot(charsa[(charsa['jdate'].dt.year>2000) & (charsa['jdate'].dt.year<=2020)][charsa['char_port']==5].groupby(['jdate'])['rank_%s'%col].count(),label='port5')
#     plt.title('%s_num_a'%col)
#     plt.legend()
#     plt.savefig('./hkg2000/%s_a.jpg'%col)
#     plt.close('all')
# #     plt.show()
    
# def plot3_q(col):
#     chars['test'] = chars['rank_%s'%col]
#     chars['char_port'] = 1 + np.where(chars['test']>-0.6,1,0) + np.where(chars['test']>-0.2,1,0) + np.where(chars['test']>0.2,1,0) + np.where(chars['test']>0.6,1,0)
#     vwret = chars.groupby(['jdate', 'char_port']).apply(wavg, 'retm', 'lag_me').to_frame().reset_index().rename(columns={0: 'vwret'})

#     vwmkt = chars.groupby(['jdate']).apply(wavg, 'retm', 'lag_me').to_frame()
#     vwmkt = vwmkt.reset_index()
#     vwmkt['jdate'] = pd.to_datetime(vwmkt['jdate'])

#     # vwret['jdate'] = pd.to_datetime(vwret['jdate'])

#     vwret = vwret[vwret['jdate'].dt.year>=2001] # maybe modify

#     # figure 1 cumsum ret
# #     plt.cla()
    
#     plt.figure(figsize=(15,5), dpi=80)
#     plt.figure(1)
#     plt.clf()
#     plt.subplot(131)
#     l1 = plt.plot(vwret[vwret['char_port'] == 1]['jdate'], vwret[vwret['char_port'] == 1]['vwret'].cumsum(), label='port1')
#     l2 = plt.plot(vwret[vwret['char_port'] == 2]['jdate'], vwret[vwret['char_port'] == 2]['vwret'].cumsum(), label='port2')
#     l3 = plt.plot(vwret[vwret['char_port'] == 3]['jdate'], vwret[vwret['char_port'] == 3]['vwret'].cumsum(), label='port3')
#     l4 = plt.plot(vwret[vwret['char_port'] == 4]['jdate'], vwret[vwret['char_port'] == 4]['vwret'].cumsum(), label='port4')
#     l5 = plt.plot(vwret[vwret['char_port'] == 5]['jdate'], vwret[vwret['char_port'] == 5]['vwret'].cumsum(), label='port5')
#     mkt = plt.plot(vwmkt['jdate'], vwmkt[0].cumsum(), label='mkt')
#     plt.title('rank_%s_q'%col)
#     plt.legend()
#     # figure 2
#     # plt.figure(132)
#     plt.subplot(132)
#     plt.bar('port1',vwret[vwret['char_port']==1]['vwret'].mean(),label='port1')
#     plt.bar('port2',vwret[vwret['char_port']==2]['vwret'].mean(),label='port2')
#     plt.bar('port3',vwret[vwret['char_port']==3]['vwret'].mean(),label='port3')
#     plt.bar('port4',vwret[vwret['char_port']==4]['vwret'].mean(),label='port4')
#     plt.bar('port5',vwret[vwret['char_port']==5]['vwret'].mean(),label='port5')
#     plt.bar('mkt',vwret['vwret'].mean())
#     plt.title('%s_avg_ret_q'%col)
#     plt.legend()
#     # figure 3
#     # plt.figure(133)
#     plt.subplot(133)
#     plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==1].groupby(['jdate'])['rank_%s'%col].count(),label='port1')
#     plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==2].groupby(['jdate'])['rank_%s'%col].count(),label='port2')
#     plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==3].groupby(['jdate'])['rank_%s'%col].count(),label='port3')
#     plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==4].groupby(['jdate'])['rank_%s'%col].count(),label='port4')
#     plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2020)][chars['char_port']==5].groupby(['jdate'])['rank_%s'%col].count(),label='port5')
#     plt.title('%s_num_q'%col)
#     plt.legend()
#     plt.savefig('./hkg2000/%s_q.jpg'%col)
#     plt.close('all')
# #     plt.show()


# plotlist_q = ['bm', 'ep', 'cp', 'agr', 'alm', 'ato', 'cash', 'cashdebt', 
#               'chpm', 'chtx', 'cinvest', 'depr', 'gma', 'grltnoa', 'lev', 
#               'lgr', 'nincr', 'noa', 'op', 'mom12m', 'mom36m', 'mom60m', 
#               'mom6m', 'mom1m', 'sgr', 'rna', 'roa', 'roe', 'rsup', 'seas1a', 'sp', 'acc', 'pctacc', 'pm']
# for char in plotlist_q:
#     plot3_q(char)


# plotlist_a = ['bm', 'ep', 'cp', 'agr', 'ato', 'cash', 'cashdebt', 
#               'chpm', 'chtx', 'depr', 'gma', 'grltnoa', 'lev', 
#               'lgr', 'noa', 'op', 'mom12m', 'mom36m', 'mom60m', 
#               'mom6m', 'mom1m', 'sgr', 'rna', 'roa', 'roe', 'rsup', 'seas1a', 'sp', 'acc', 'pctacc', 'pm']
# for char in plotlist_a:
# 	plot3_a(char)




# for i in com_list:
# 	if secm[secm['gvkey']==i].empty:
# 		print(i)





'''