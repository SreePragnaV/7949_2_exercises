#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


sales = pd.read_excel('SaleData.xlsx')


# In[3]:


sales.head()


# In[4]:


sales['Item'].unique()


# In[5]:


sales.info()


# In[6]:


groupItem = sales.groupby('Item').min()


# In[7]:


groupItem['Sale_amt']


# In[8]:


#Q1
def least_sales(df):
    groupItem = df.groupby('Item').min()
    grItem = groupItem['Sale_amt'].reset_index()
    return grItem
ls = least_sales(sales)
ls


# In[9]:


#Q2
def sales_year_region(df):
    gYR = df.groupby(['Year','Region']).sum()
    return gYR

sales['Year'] = sales['OrderDate'].apply(lambda n: n.year)
#sales.tail()
groupYR = sales_year_region(sales)
groupYR


# In[10]:


#Q3
import datetime as dt
from dateutil.relativedelta import relativedelta
from datetime import date

def days_diff(df):
    t3 = pd.Timestamp('today')
    td = t3 - df['OrderDate']
    df['no of days'] = td/np.timedelta64(1,'D')
    return df

s = days_diff(sales)
s


# In[11]:


#Q4
def mrsls(dfm):
    return sales[sales['Manager']==dfm]['SalesMan'].unique()
def mgr_slsmn(df):
    g = pd.DataFrame(df['Manager'].unique(),columns=['mngr'])
    g['Slsmn']= g['mngr'].apply(lambda x:mrsls(x))
    return g
mngr_sls = mgr_slsmn(sales)
mngr_sls


# In[12]:


#Q5
def slsmn_units(df):
    groupRUnits = df.groupby('Region').sum()
    gR1 = groupRUnits['Units'].copy()
    groupRSM = df.groupby('Region')['SalesMan'].nunique()
    groupRUSM = pd.concat([groupRSM, gR1], axis=1)
    return groupRUSM
groupRUnitsSlmn = slsmn_units(sales)
groupRUnitsSlmn


# In[13]:


#Q6
def sales_pct(df):
    groupMS = df.groupby('Manager')['Sale_amt'].sum() 
    groupms = pd.concat([groupMS],axis=1)
    groupms['Sale_amt'] = groupms['Sale_amt'].apply(lambda n: (n/sum(groupMS))*100)
    return groupms

groupMSales = sales_pct(sales)
groupMSales


# In[14]:


imd = pd.read_csv('imdb.csv',escapechar='\\')


# In[15]:


imd.head()


# In[16]:


#Q7
def fifth_movie(df):
    return df['imdbRating'].iloc[4]
fifth_movie_rating = fifth_movie(imd)
fifth_movie_rating


# In[17]:


#Q8
highest_runtime = imd['duration'].max()
lowest_runtime = imd['duration'].min()
print('Highest and lowest duration run times are '+str(highest_runtime)+', '+str(lowest_runtime))


# In[18]:


#Q9
def sort_df(df):
    sortRD = df.sort_values(['year','imdbRating'], ascending=[True, False])
    return sortRD
sortYearRating = sort_df(imd)
sortYearRating.head(30)


# In[19]:


mvemeta = pd.read_csv('movie_metadata.csv',escapechar='\\')


# In[20]:


mvemeta.info()


# In[21]:


#Q10
def subset_df(df):
    movie = df[(df['duration']>30) & (df['duration']<180)]
    movie = movie[(movie['gross']>2000000) & (movie['budget']<1000000)]
    return movie

movieSubset = subset_df(mvemeta)
movieSubset


# In[22]:


dia = pd.read_csv('diamonds.csv')


# In[23]:


len(dia.index)


# In[24]:


dia.head()


# In[25]:


#Q11
def dupl_rows(df):
    dupdia = dia[dia.duplicated()]
    return len(dupdia.index)

noDupDia = dupl_rows(dia)
print('Number of duplicate rows are '+str(noDupDia))


# In[26]:


#Q12
def drop_row(df):
    df = df.dropna(axis=0, subset=['carat','cut'])
    return df
dropCarCut = drop_row(dia)
len(dropCarCut.index)


# In[27]:


#Q13
def sub_numeric(df):
    dfS = df._get_numeric_data()
    return dfS
dfS = sub_numeric(dia)
dfS.info()


# In[28]:


#Q14
def volume(df):
    dia1 = df.dropna(axis=0, subset=['z'])
    dia1 = dia1[dia1['z'] != 'None']
    dia1['zFloat'] = dia1['z'].astype(float)
    dia2 = dia1[dia1['depth']<=60]
    dia3 = dia1[dia1['depth']>60]
    dia2['volume'] = 8
    dia3['volume'] = dia3['x']*dia3['y']*dia3['zFloat']
    dia4 = pd.concat([dia2, dia3])
    return dia4

diaV = volume(dia)
diaV


# In[29]:


#Q15
def impute(df):
    df['price'].fillna(value=dia['price'].mean(),inplace=True)
impute(dia)
dia


# In[30]:


imd2 = pd.read_csv('imdb_2.csv',escapechar='\\')


# In[31]:


#Q16
import timeit
imd2['imdbRating'] = imd2['imdbRating'].astype(float)
imd2combo = pd.DataFrame(imd2.groupby(['year','type'])['imdbRating'].mean().reset_index())
imd2combo['MaximumRating'] = pd.DataFrame(imd2.groupby(['year','type'])['imdbRating'].max().reset_index())['imdbRating']
imd2combo['MinimumRating'] = pd.DataFrame(imd2.groupby(['year','type'])['imdbRating'].min().reset_index())['imdbRating']
imd2combo['TotalRunTime'] = pd.DataFrame(imd2.groupby(['year','type'])['duration'].sum().reset_index())['duration']
imd2combo['genrecombo'] = 'none'
def inlist(l,s):
    if s in l:
        return 0
    else:
        return 1
def listcombo(i):
    imd3 = imd2[(imd2['year']==imd2combo['year'].iloc[i,]) & (imd2['type']==imd2combo['type'].iloc[i,])]
    im4 = imd3.iloc[:,16:43]
    l = []
    for j in range(len(im4.index)):
        for k in range(0,27):
            if im4.iloc[j,k]>0:
                p = inlist(l,imd3.columns[k+16])
                if p==1:
                    l.append(imd3.columns[k+16])
    return l
start = timeit.default_timer()
for i in range(len(imd2combo.index)):
    imd2combo.iat[i,6] = listcombo(i)
stop = timeit.default_timer()
imd2combo
#print(stop-start)


# In[32]:


#Q17

import timeit
imd2['movielen'] = imd2['title'].apply(len)
imd2['movielen'] = imd2['movielen'].apply(lambda x: x-7)
imd2qu = pd.DataFrame(imd2.groupby(['year'])['movielen'].max().reset_index())
imd2qu.rename(columns={"movielen": "max_length"}, inplace=True)
imd2qu[' min_length'] = pd.DataFrame(imd2.groupby(['year'])['movielen'].min().reset_index())['movielen']
imd2qu[' num_videos_less_than25Percentile'] = 0
imd2qu['num_videos_25_50Percentile'] = 0
imd2qu['num_videos_50_75Percentile'] = 0
imd2qu['num_videos_greaterthan75Precentile '] = 0

def less25qu(i):
    im4 = imd2[imd2['year']==imd2qu['year'].iloc[i,]]
    qu1 = np.percentile(imd2.movielen, 25)
    return np.sum(im4['movielen']<qu1)
def qu25_50(i):
    im4 = imd2[imd2['year']==imd2qu['year'].iloc[i,]]
    qu1 = np.percentile(imd2.movielen, 25)
    qu2 = np.percentile(imd2.movielen, 50)
    return len(im4[(im4['movielen']>=qu1) & (im4['movielen']<qu2)])
def qu50_75(i):
    im4 = imd2[imd2['year']==imd2qu['year'].iloc[i,]]
    qu2 = np.percentile(imd2.movielen, 50)
    qu3 = np.percentile(imd2.movielen, 75)
    return len(im4[(im4['movielen']>=qu2) & (im4['movielen']<qu3)])
def great75qu(i):
    im4 = imd2[imd2['year']==imd2qu['year'].iloc[i,]]
    qu3 = np.percentile(imd2.movielen, 75)
    return np.sum(im4['movielen']>=qu3)

for i in range(len(imd2qu.index)):
    imd2qu.iat[i,3] = less25qu(i)
    imd2qu.iat[i,4] = qu25_50(i)
    imd2qu.iat[i,5] = qu50_75(i)
    imd2qu.iat[i,6] = great75qu(i)
imd2qu
imd2 = imd2.dropna(axis=0, subset=['year'])

start = timeit.default_timer()
def f(x):
    if x['movielen'] < np.percentile(imd2.movielen, 25): return 1
    elif (x['movielen'] >= np.percentile(imd2.movielen, 25)) and (x['movielen'] < np.percentile(imd2.movielen, 50)): return 2
    elif (x['movielen'] >= np.percentile(imd2.movielen, 50)) and (x['movielen'] < np.percentile(imd2.movielen, 75)): return 3
    elif x['movielen'] >= np.percentile(imd2.movielen, 75): return 4
imd2['quartile'] = imd2.apply(f, axis=1)
y = imd2["year"].astype(str).to_numpy() 
q = imd2["quartile"].astype(str).to_numpy()
imd2quct = pd.crosstab(y, q, rownames=['year'], colnames=['quartile'])
stop = timeit.default_timer()
imd2quct
imd2qu
imd2quct2 = pd.pivot_table(imd2, index='year', columns='quartile',values='imdbRating',aggfunc='mean',fill_value=0)
imd2quct2
# print(stop-start)


# In[33]:


imd2quct


# In[34]:


imd2qu


# In[35]:


#Q18
diaeq = pd.qcut(diaV['volume'], q=7
             , duplicates='drop')
diaeq.value_counts()
bins = diaeq.astype(str).to_numpy() 
cut = diaV["cut"].to_numpy()
crossbinscut = pd.crosstab(bins, cut, rownames=['bins'], colnames=['cut'])
crossbinscut
total = np.sum(crossbinscut.values)
#crossbinscut = pd.crosstab(bins, cut, rownames=['bins'], colnames=['cut']).apply(lambda r:(r/r.sum())*100,axis=1)
crossbinscut = pd.crosstab(bins, cut, rownames=['bins'], colnames=['cut']).apply(lambda r:(r/total)*100,axis=1)
crossbinscut


# In[36]:


mvemeta2 = pd.read_csv('movie_metadata.csv',escapechar='\\')


# In[37]:


#Q19
movieyear = pd.DataFrame(mvemeta2['title_year'].unique())
movieyear.rename(columns={0: "year"}, inplace=True)
yearsort = movieyear.sort_values(['year'], ascending=[0])[:10]
yr_srt_qu = pd.qcut(yearsort['year'], q=4, duplicates='drop')
yr_df1 = pd.DataFrame(yr_srt_qu.value_counts()).reset_index()
yr_df1.rename(columns={"index": "period","year":"count"}, inplace=True)
yr_srt_qu2 = pd.qcut(yearsort['year'], q=4, duplicates='drop',labels=False)
yr_df2 = pd.DataFrame(yr_srt_qu2.value_counts()).reset_index()
yr_df2.rename(columns={"index": "qrtr","year":"count"}, inplace=True)
yr_df3 = pd.concat([yr_df1['period'], yr_df2['qrtr']], axis=1)
yr_df3
yearsort["qrt"] = 5
def qr(yr):
    for i in range(len(yr_df3.index)):
        if yr in yr_df3["period"][i]:
            return yr_df3["qrtr"][i]
        
for i in yearsort.index:
    yearsort["qrt"][i] = qr(yearsort["year"][i])
yearsort = pd.DataFrame(yearsort.groupby('qrt')['year'].apply(list)).reset_index()

mvemeta2['qrtr'] = 5
def findqrtr(yr):
    for i in range(len(yearsort.index)):
        if yr in yearsort['year'][i]:
            return yearsort['qrt'][i]
for i in range(len(mvemeta2.index)):
    mvemeta2['qrtr'][i] = findqrtr(mvemeta2['title_year'][i])
mvemeta2[['title_year','qrtr']]
mvert = mvemeta2.groupby('qrtr')['imdb_score'].mean().reset_index()
mvert['count'] = pd.DataFrame(mvemeta2.groupby(['qrtr'])['movie_imdb_link'].count().reset_index())['movie_imdb_link']

l = []
def inlist(ij,l):
    if ij in l:
        return 0
    return 1
def get_genlist(df):
    for i in range(len(df.index)):
        k = []
        k = df['genres'][i].split('|')
        for j in k:
            p = inlist(j,l)
            if p==1:
                l.append(j)
    return l
l2 = get_genlist(mvemeta2)
l2 #list of all genres
for i in l2:
    mvert[i] = 0
    mvemeta2[i] = 0
mvert
mvemeta2

for i in range(len(mvemeta2.index)):
    k = []
    k = mvemeta2['genres'][i].split('|')
    for j in k:
        mvemeta2[j][i] = 1

for i in range(len(mvert.index)):
    mvesub = mvemeta2[mvemeta2['qrtr']==mvert['qrtr'][i]]
    mvesub2 = mvesub.sort_values(by=['gross'])
    r = int(round(mvert['count'][i]/10))
    mvesub3 = mvesub2.head(r)
    mvdf = mvesub3.sum(axis = 0)[25:]
    mvert.iloc[i,3:] = mvdf
mvert


# In[38]:


#Q20
imdDur1 = pd.qcut(imd2['duration'], q=10, duplicates='drop')
imdDur2 = pd.qcut(imd2['duration'], q=10, duplicates='drop',labels =False)
imdDur_df1 = pd.DataFrame(imdDur1.value_counts()).reset_index()
imdDur_df1.rename(columns={"index":"durationperiod","duration":"count"},inplace=True)
imdDur_df2 = pd.DataFrame(imdDur2.value_counts()).reset_index()
imdDur_df2.rename(columns={"index":"dec","duration":"count"},inplace=True)
imdDur_df3 = pd.concat([imdDur_df1['durationperiod'], imdDur_df2['dec']], axis=1)

def find_decile(dur):
    for i in range(len(imdDur_df3)):
        if dur in imdDur_df3['durationperiod'][i]:
            return imdDur_df3['dec'][i]
imd2['durDecile'] = 11
for i in imd2.index:
    imd2['durDecile'][i] = find_decile(imd2['duration'][i])
imdDurG = pd.DataFrame(imd2.groupby('durDecile')['nrOfNominations'].sum().reset_index())
imdDurG['WinsCount'] = pd.DataFrame(imd2.groupby('durDecile')['nrOfWins'].sum().reset_index())['nrOfWins']
imdDurG['Count'] = pd.DataFrame(imd2.groupby('durDecile').count().reset_index())['tid']

agg_gen = pd.DataFrame(imd2.groupby(['durDecile']).agg({imd2.columns[16]:sum,imd2.columns[17]:sum,imd2.columns[18]:sum,imd2.columns[19]:sum,imd2.columns[20]:sum,imd2.columns[21]:sum,imd2.columns[22]:sum,imd2.columns[23]:sum,imd2.columns[24]:sum,imd2.columns[25]:sum,imd2.columns[26]:sum,imd2.columns[27]:sum,imd2.columns[28]:sum,imd2.columns[29]:sum,imd2.columns[30]:sum,imd2.columns[31]:sum,imd2.columns[32]:sum,imd2.columns[33]:sum,imd2.columns[34]:sum,imd2.columns[35]:sum,imd2.columns[36]:sum,imd2.columns[37]:sum,imd2.columns[38]:sum,imd2.columns[39]:sum,imd2.columns[40]:sum,imd2.columns[41]:sum,imd2.columns[42]:sum,imd2.columns[43]:sum})).reset_index()

def gen1(i):
    max1 = 0
    max1G = "n1"
    for col in agg_gen:
        if col != "durDecile":
            if agg_gen[col][i]>max1:
                max1 = agg_gen[col][i]
                max1G = col
    return max1G
def gen2(i,maxg1):
    max2 = 0
    max2G = "n2"
    for col in agg_gen:
        if col!="durDecile" and col!=maxg1:
            if agg_gen[col][i]>max2:
                max2= agg_gen[col][i]
                max2G= col
    return max2G
def gen3(i,maxg1,maxg2):
    max3= 0
    max3G= "n2"
    for col in agg_gen:
        if col!="durDecile" and col!=maxg1 and col!=maxg2:
            if agg_gen[col][i]>max3:
                max3=agg_gen[col][i]
                max3G=col
    return max3G

imdDurG['genre1'] = 'n1'
imdDurG['genre2'] = 'n2'
imdDurG['genre3'] = 'n3'
for i in range(len(imdDurG.index)):
    imdDurG['genre1'][i] = gen1(i)
    imdDurG['genre2'][i] = gen2(i,gen1(i))
    imdDurG['genre3'][i] = gen3(i,gen1(i),gen2(i,gen1(i)))

imdDurG


# In[ ]:




