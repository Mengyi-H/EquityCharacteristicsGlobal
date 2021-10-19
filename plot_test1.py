import pickle as pkl
import pyarrow.feather as feather
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import datetime
from pandas.tseries.offsets import *

# with open('chars_a_rank.feather', 'rb') as f:
#     charsa = feather.read_feather(f)

with open('chars_q_rank_hkg.feather', 'rb') as f:
    chars = feather.read_feather(f)
chars['date_fn'] = chars['jdate'] + QuarterEnd(0)
with open('chars_q_hkg.feather', 'rb') as f:
    fqsm = feather.read_feather(f)
fqsm['date_fn'] = fqsm['jdate'] + QuarterEnd(0)
    
def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan

def plot4_q(col,chars,fqsm):
    chars = chars.copy()
    chars = chars.dropna(subset = ['rank_acc'])
    chars['test'] = chars['rank_%s'%col]
    #chars['char_port'] = 1 + np.where(chars['test']>-0.6,1,0) + np.where(chars['test']>-0.2,1,0) + np.where(chars['test']>0.2,1,0) + np.where(chars['test']>0.6,1,0)
    chars = chars.drop(chars[chars['retm']>0.2].index)
    chars = chars.drop(chars[chars['retm']<-0.2].index)
    chars['char_port'] = 1 + np.where(chars['test']>-0.6,1,0) + np.where(chars['test']>-0.2,1,0) + np.where(chars['test']>0.2,1,0) + np.where(chars['test']>0.6,1,0)

    vwret = chars.groupby(['jdate', 'char_port']).apply(wavg, 'retm', 'lag_me').to_frame().reset_index().rename(columns={0: 'vwret'})
    vwmkt = chars.groupby(['jdate']).apply(wavg, 'retm', 'lag_me').to_frame()
    vwmkt = vwmkt.reset_index()
    vwmkt['jdate'] = pd.to_datetime(vwmkt['jdate'])
    # vwret['jdate'] = pd.to_datetime(vwret['jdate'])
    vwret = vwret[vwret['jdate'].dt.year>=2000] # maybe modify
    # vwret['cumret'] = 1
    # for i in range(1,6):
    #     temp_index = list(vwret[vwret['char_port']==i].index)
    #     for j in range(0,len(temp_index)-1):
    #         vwret['cumret'][temp_index[j+1]] = vwret['cumret'][temp_index[j]] * (1+vwret['vwret'][temp_index[j+1]])
    #         print(j)
    # figure 1 cumsum ret
    #     plt.cla()
    plt.figure(figsize=(30,20), dpi=100)
    plt.figure(1)
    plt.clf()
    plt.subplot(221)
    l1 = plt.plot(vwret[vwret['char_port'] == 1]['jdate'], (vwret[vwret['char_port'] == 1]['vwret']+1).cumprod(), label='port1')
    l2 = plt.plot(vwret[vwret['char_port'] == 2]['jdate'], (vwret[vwret['char_port'] == 2]['vwret']+1).cumprod(), label='port2')
    l3 = plt.plot(vwret[vwret['char_port'] == 3]['jdate'], (vwret[vwret['char_port'] == 3]['vwret']+1).cumprod(), label='port3')
    l4 = plt.plot(vwret[vwret['char_port'] == 4]['jdate'], (vwret[vwret['char_port'] == 4]['vwret']+1).cumprod(), label='port4')
    l5 = plt.plot(vwret[vwret['char_port'] == 5]['jdate'], (vwret[vwret['char_port'] == 5]['vwret']+1).cumprod(), label='port5')
    mkt = plt.plot(vwmkt['jdate'], (vwmkt[0]+1).cumprod(), label='mkt')
    plt.title('Cumulative Return: %s_q'%col)
    plt.legend()
    
    # figure 2
    # plt.figure(132)
    plt.subplot(222)
    plt.bar('port1',vwret[vwret['char_port']==1]['vwret'].mean(),label='port1')
    plt.bar('port2',vwret[vwret['char_port']==2]['vwret'].mean(),label='port2')
    plt.bar('port3',vwret[vwret['char_port']==3]['vwret'].mean(),label='port3')
    plt.bar('port4',vwret[vwret['char_port']==4]['vwret'].mean(),label='port4')
    plt.bar('port5',vwret[vwret['char_port']==5]['vwret'].mean(),label='port5')
    plt.bar('mkt',vwret['vwret'].mean())
    plt.title('Average Return: %s_q'%col)
    
     # figure 3
    # plt.figure(133)
    plt.subplot(223)
    plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2021)][chars['char_port']==1].groupby(['date_fn'])['rank_%s'%col].count()[0:-1],label='port1')
    plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2021)][chars['char_port']==2].groupby(['date_fn'])['rank_%s'%col].count()[0:-1],label='port2')
    plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2021)][chars['char_port']==3].groupby(['date_fn'])['rank_%s'%col].count()[0:-1],label='port3')
    plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2021)][chars['char_port']==4].groupby(['date_fn'])['rank_%s'%col].count()[0:-1],label='port4')
    plt.plot(chars[(chars['jdate'].dt.year>2000) & (chars['jdate'].dt.year<=2021)][chars['char_port']==5].groupby(['date_fn'])['rank_%s'%col].count()[0:-1],label='port5')
    temp1 = pd.DataFrame(fqsm.groupby(['date_fn'])[['%s'%col,'gvkey']].count()).reset_index()
    temp1['missing'] = temp1['gvkey']-temp1['%s'%col]
    plt.plot(temp1['date_fn'][0:-1],temp1['missing'][0:-1],label='missing')
    plt.title('Firm Numbers : %s_q'%col)
    plt.legend()
    # figure 4
    #
    plt.subplot(224)
    plt.plot(temp1['date_fn'][0:-1], temp1['%s'%col][0:-1],label='char')
    plt.plot(temp1['date_fn'][0:-1], temp1['gvkey'][0:-1],label='market')
    plt.title('firm numbers')
    plt.legend()
    plt.savefig('./hkgplot_test/%s_q.jpg'%col)
    plt.close('all')
    vwret.to_csv('./hkgplot_test/%s_q.csv'%col)





plotlist_q = ['bm', 'ep', 'cp', 'agr', 'alm', 'ato', 'cash', 'cashdebt', 
              'chpm', 'chtx', 'cinvest', 'depr', 'gma', 'grltnoa', 'lev',
              'lgr', 'nincr', 'noa', 'op', 'mom12m', 'mom36m', 'mom60m', 
              'mom6m', 'mom1m', 'sgr', 'rna', 'roa', 'roe', 'rsup', 'seas1a', 'sp', 'acc', 'pctacc', 'pm','me']
for char in plotlist_q:
    plot4_q(char,chars,fqsm)


