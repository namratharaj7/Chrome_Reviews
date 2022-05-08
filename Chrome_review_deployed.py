#!/usr/bin/env python
# coding: utf-8

# In[86]:


# importing necessary libraries & packages
import streamlit as st
# defining the title of the web app
st.title("Chrome Reviews having Positive Text Reviews with Negative Ratings")
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from textblob import TextBlob



# In[87]:
import os
os.chdir('C:\\Users\\Namratha')

#reading the csv file of the chrome_review dataset
df =  pd.read_csv('C:\\Users\\Namratha\\Desktop\\datasets\\archive (1)\\chrome_reviews.csv')

# In[88]:


#check if there are null entries in any column
df.info()


# In[89]:


#Let us consider only the 'Text' & 'Star' columns for our training 
df = df[['Text','Star']]
#read the 'Text' column as string
df['Text'] = df['Text'].astype(str)


# In[90]:


#defining the function to return the sentiment polarity score of the text review using textblob library
def find_pol(text):
    return TextBlob(text).sentiment.polarity
#applying the above function on the 'Text' column & saving the sentiment polarity value in a new column named 'S_polarity'
df['S_polarity'] = df['Text'].apply(find_pol)


# **Note**:
# 
# *The value of S_polarity lies between -1 & 1.*
# 
# *S_polarity value <0 indicates a Negative sentiment*,
# 
# *S_polarity value >0 indicates a Positive sentiment*,
# 
# *S_polarity value =0 indicates a Neutral sentiment*
# 
# example:
# *S_polarity value = -1 indicates  Most Negative sentiment(100% Negative)*,
# 
# *S_polarity value = 1 indicates Most Positive sentiment(100% Positive)*

# In[91]:


#defining another sentiment function using SentimentIntensityAnalyzer from nltk library to better achieve the desired result
def sentiment_analyse(text):
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    if score['neg'] > score['pos']:
            return "Negative"
    elif score['neg'] < score['pos']:
            return "Positive"
    else:
            return "Neutral"
#applying the above function on the 'Text' column & saving the sentiment in a new column named 'Sentiment'
df['Sentiment']=df['Text'].apply(sentiment_analyse)


# #### 6. Obtaining the data of reviews with good text review & bad rating
# Let us consider that the star ratings of (1,2) are bad, 3 is neutral and (4,5) is good.
# Since we want to find the reviews with bad star rating(1,2) & good/positive sentiment,
# let us use the following 3 parameters to obtained the desired result:
# 
# a. To filter bad ratings: **(df['Star'] <3)**
# 
# b. To filter positive sentiment from *find_pol* function: 
# 
#    Since we know that (S_polarity value >0) indicates a Positive sentiment, let us filter text review with positiveness of 60% & above. i.e.,
#    
# **(df['S_polarity'] >= 0.6)**
# 
# c. To filter positive sentiment from *sentiment_analyse* function: **(df['Sentiment'] == 'Positive')**

# In[93]:


#combining all the paramters we can get the dataframe of the discrepency reviews as:
df_discrepency_reviews = df[(df['S_polarity'] >= 0.6) & (df['Star'] <3) & (df['Sentiment']=='Positive')]


# In[96]:


#Code for Deployment of the model in Streamlit


# In[101]:


# Creating to CSV file upload option
uploaded_file = st.file_uploader("Choose a CSV file for checking review-rating discrepancy",type=["csv"])

# Performing the earlier steps performed in training file again for the uploaded test csv file
if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    df_test['Text'] = df_test['Text'].astype(str)
    df_test['Sentiment']=df_test['Text'].apply(sentiment_analyse)
    df_test['S_polarity'] = df_test['Text'].apply(find_pol)
    df_test_discrepency_reviews = df_test[(df_test['S_polarity'] >= 0.6) & (df_test['Star'] <3) & (df_test['Sentiment']=='Positive')]
    # Display of reviews having good text reviews & negative ratings on web app using streamlit
    st.write("The list of reviews data having reviews and ratings mismatch:")
    st.write(df_test_discrepency_reviews)
