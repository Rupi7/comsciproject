#!/usr/bin/env python
# coding: utf-8

# In[31]:



#this is a sentiment analysis program of amazon reviews
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
#I need panadas, numpy and nltk to analyze and help make arranging the numbers easier
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
#i sorted the wilso bound column for an easier analysis analysis
df.drop("Unnamed: 0", inplace=True, axis=1)
#formatting it in a certain way so itll be easier to read
#gettingt he data set
df.head()
#printing the dataset


# In[32]:


def misValAnalysis(df):
    #creating a function for missing value analysis
    columns= [col for col in df.columns if df[col].isnull().sum()>0]
    #getting the null colummns
    misscol=df[columns].isnull().sum.sort_values(ascending=True)
    #creating a variable for missing columns
    ratio=(df[columns].isnull().sum/df.shape[0]*100).sort_values(ascending=True)
    #ratio of all values
    missingdf=pd.concat([misscol,np.round(ratio,2)], axis=1, keys=['missing values','Ratio'])
    missingdf=pd.DataFrama(missingdf)
    return missingdf
#function for missinh values analysis

#function to check data fram
def checkdataframe (df, head=5, tail=5):
    #creating a function to check the dataframe
    
    print("Shape".center(82,'~'))
    #using ~ cuz it looks nice
    print('Rows: {}'.format(df.shape[0]))
    #formating the values
    print('columns:{}'.format(df.shape[1]))
    print("Types".center(82,'~'))
    print(df.dtypes)
    print("".center(82,,'~'))
    #placing the positions
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
#choosing colours for the graph
#choosing colours
def catagoricalVariSum(df,column_name):
    #creating a functionm to catagorize the variable summeries
    fig=makeSubplots(rows = 1, cols = 2,
                    subplotTitle=('CountPlot','Percentage'),
                    specs=[[{'types': 'xy'},{'types':'domain'}]])
    #variable fig is essentially the subplots
    fig.add_trace(go.Bar(y = df[column_name].value_count().values.tolist(),
                       x= [str[i] for i in df[column_name].value_counts().index)]. text=df.value_counts().values.tolist(), textfont = dict(size=14),
                       name = column_name,
                       textposition='auto'
                       showLegend = False,
                       marker= dict(color=constraints,
                                    line = dict(colour='#9D8189', width = 1))),
                row=1, col=1)
    #making th eresult look nice
    fig.add_trace(go.Pie(labels=df[column_name].value_counts().keys(),
                 values= df[colum_name].value_counts().values,
                         #got this part from stack overflow
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
    #leanred why this function works from my dad


# In[ ]:


catagoricalVariSum(df,'overall')


# In[ ]:


df.reviewText.head()
reviewExample=df.reviewText[2031]
print(reviewExample)
reviewExample=re.sub("[^a-zA-Z]","", reviewExample)
#getting rif of of punctutatuon so itll be easier for he machine to comprehend
print(reviewExample)
reviewExample=reviewExample.lower().split()
print(reviewExample)
#making it all lower so th eprogram wonht mistake words starting with capitals as a different word
#Its gonna be one word per line since we are splitting it
rt=lambda x: re.sub("[^a-zA-Z]", ' ', str(x))
df['reviewText']= df['reviewText'].map(rt)
df['reviewText']= df['reviewText'].str.lower()
df.head()


# In[ ]:


#going to do the sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
df=[['polararity','subjectivity']] = df['reviewText'].apply()(lambda Text:pd.series(TextBlob(Text).sentiment))
#analyzing sentiment

for inder, row in df['reviewText'].iteritems():
    #creating an iterartor of the values
    score=SentimentIntensityAnalyzer.polarity_scores(row)
    #getting the score
    neg=score['negative']
    #negative scores
    neu= score['neutral']
    #neutral scores
    #positive scores
    pos=score['positive']
    if neg>pos:
        df.loc[index,'sentiment'] ='Negative'
    elif pos>neg:
         df.loc[index,'sentiment']='Positive'
    else:
        df.loc[index,'sentiment']='Neutral'
        
df[df['sentiment']=='Positive'].sort_values('wilson_lower_bound',
                                           ascending=False).head(5)
#sortiing the values
catagoricalVariSum(df,'sentiment')
#final result printing



#https://www.ahmedbesbes.com/blog/end-to-end-machine-learning
#https://docs.aws.amazon.com/comprehend/latest/dg/how-sentiment.html
#https://www.kaggle.com/datasets/bittlingmayer/amazonreviews
#https://www.youtube.com/watch?v=G6TbcyFxrms&ab_channel=GreatLearning
#https://thecleverprogrammer.com/2021/07/20/amazon-product-reviews-sentiment-analysis-with-python/
#https://medium.com/@eesraerkan/sentiment-analysis-of-product-reviews-on-amazon-com-9bde346519b8
#https://ashleygingeleski.com/2021/03/31/sentiment-analysis-of-product-reviews-with-python-using-nltk/
#alot of my code was helped by my dad and i learned and understood why my program does what it does from simplilearn's amaazon analysis video, some other bits of my code are from stack overflowe

