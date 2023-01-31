#!/usr/bin/env python
# coding: utf-8

# In[31]:



#this is a sentiment analysis program of amazon reviews
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import seaborn as sns
import cufflinks as cf
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
cf.go_offline();
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings('ignore')
warnings.warn('This will not show')
pd.set_option('display.max_columns', None)

#importing nessecary modules


# In[39]:


df= pd.read_csv("amazon.csv")
df=df.sort_values("wilson_lower_bound", ascending = False)
df.drop("Unnamed: 0", inplace=True, axis=1)
#gettingt he data set
df.head()


# In[32]:


def misValAnalysis(df):
    columns= [col for col in df.columns if df[col].isnull().sum()>0]
    misscol=df[columns].isnull().sum.sort_values(ascending=True)
    ratio=(df[columns].isnull().sum/df.shape[0]*100).sort_values(ascending=True)
    missingdf=pd.concat([misscol,np.round(ratio,2)], axis=1, keys=['missing values','Ratio'])
    missingdf=pd.DataFrama(missingdf)
    return missingdf
#function for missinh values analysis

#function to check data fram
def checkdataframe (df, head=5, tail=5):
    
    print("Shape".center(82,'~'))
    print('Rows: {}'.format(df.shape[0]))
    print('columns:{}'.format(df.shape[1]))
    print("Types".center(82,'~'))
    print(df.dtypes)
    print("".center(82,,'~'))
    print(misValAnalysis(df))
    print('duplicated_values'..center(83,'~'))
    print(df.duplicated().sum())
    print("Quantiles".center(83,'~'))
    print(df.quantile([0,0.05,0.50,0.95,0.99,1]).T)
    
checkdataframe(df)


# In[40]:


def checkclassframe(dataframe):
    ndf=pd.DataFrame({'variable':dataFrame.columns,
                     'Classes':[dataFrame[i].nunique() \ for i in dataFrame.columns]})
    ndf=ndf.sort_values('Classes', ascending = False)
    ndf=ndf.reset_index(drop=True)
    return ndf


# In[ ]:


constraints = ['#B34D22','#EBE00C','1FEB0C','#0C92EB','#F4ACB7']
#choosing colours
def catagoricalvariSum(df,column_name):
    fig=makeSubplots(rows = 1, cols = 2,
                    subplotTitle=('CountPlot','Percentage'),
                    specs=[[{'types': 'xy'},{'types':'domain'}]])
    fig.add_trace(go.Bar(y = df[column_name].value_count().values.tolist(),
                       x= [str[i] for i in df[column_name].value_counts().index)]. text=df.value_counts().values.tolist(), textfont = dict(size=14),
                       name = column_name,
                       textposition='auto'
                       showLegend = False,
                       marker= dict(color=constraints,
                                    line = dict(colour='#9D8189', width = 1))),
                row=1, col=1)
    fig.add_trace(go.Pie(labels=df[column_name].value_counts().keys(),
                 values= df[colum_name].value_counts().values,
                 textFont = dict(size=18),
                 texposition= 'auto',
                         showlegends=False
                         name=column_name,
                         marker=dict(colors=constraints)),
                 row = 1, col = 2)
    fig.update_layout(title={'text':column_name,
                            'y':0.9,
                           'x': 0.5,
                            'xanchor':'center',
                            'yanchor': 'top'}
                     template='plotly_white')
    iplot(fig)
    


# In[ ]:


catagoricalVariSum(df,'overall')


# In[ ]:


df.reviewText.head()
reviewExample=df.reviewText[2031]
print(reviewExample)
reviewExample=re.sub("[]")

