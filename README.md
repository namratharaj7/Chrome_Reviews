
# Chrome_reviews

The documentation is to Identify reviews with good text reviews and negative Star rating and deploying the same usind Streamlit.


## Documentation


#Identification of reviews with good text reviews and negative Star rating and deploying the same usind Streamlit.

#The code for this problem is in the Chrome_review_deployed.py

#The libraries and packages required for this problem include

- pandas (for reading file and performing operations)

- SentimentIntensityAnalyzer, vader_lexicon from NLTK (for sentiment analysis)

- TextBlob from TextBlob (for sentiment analysis)

- Streamlit (for web app deployment)

### The steps involved in solving:
    
#### 1. importing necessary libraries & packages listed above.
    
    ```bash
    import pandas as pd

    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon')

    from textblob import TextBlob

    import streamlit as st
    ```
#### 2. reading the csv file of the chrome_review dataset
```bash
  df =  pd.read_csv('Desktop/datasets/archive (1)/chrome_reviews.csv')
```
    
#### 3. analysing the data by calling .info and and checking for null values 
```bash
df.info()
```
#read the 'Text' column as string
```bash
df['Text'] = df['Text'].astype(str)
```

#### 4. defining the function to return the sentiment polarity score of the text review using textblob library
```bash
  def find_pol(text):
      return TextBlob(text).sentiment.polarity
```
#add a new column in the dataframe to save the sentiment polarity value 
```bash
df['S_polarity'] = df['Text'].apply(find_pol)
```
**Note**:

*The value of S_polarity lies between -1 & 1.*

*S_polarity value <0 indicates a Negative sentiment*,

*S_polarity value >0 indicates a Positive sentiment*,

*S_polarity value =0 indicates a Neutral sentiment*

ex:

S_polarity value = -1 indicates  Most Negative sentiment(100% Negative)
&
S_polarity value = 1 indicates Most Positive sentiment(100% Positive)

#### 5. defining another sentiment function using SentimentIntensityAnalyzer from nltk library to better achieve the desired result
```bash
def sentiment_analyse(text):
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    if score['neg'] > score['pos']:
            return "Negative"
    elif score['neg'] < score['pos']:
            return "Positive"
    else:
            return "Neutral"
```
#add a new column to the dataframe to save the sentiment obtained from the above function
```bash
df['Sentiment']=df['Text'].apply(sentiment_analyse)
```
#### 6. Obtaining the data of reviews with good text review & bad rating
> Let us consider that the star ratings of (1,2) are bad, 3 is neutral and (4,5) is good

> Since we want to find the reviews with bad star rating(1,2) & good/positive sentiment

> let us use the following 3 parameters to obtained the desired result:

a. To filter bad ratings: **(df['Star'] <3)**

b. To filter positive sentiment from *find_pol* function: **(df['S_polarity'] >= 0.6)**

c. To filter positive sentiment from *sentiment_analyse* function: **(df['Sentiment'] == 'Positive')**
   
   Here 0.6 indicates 60% & above level of positiveness in the review

#combining all the paramters we can get the dataframe of the discrepency reviews as:

```bash
df_discrepency_reviews = df[(df['S_polarity'] >= 0.6) & (df['Star'] <3) & (df['Sentiment']=='Positive')]
```
#### 7. Code for Deployment of the model in Streamlit

* #### defining the title of the web app
```bash
st.title("Chrome Reviews having Positive Text Reviews with Negative Ratings")
```
* #### Creating to CSV file upload option
```bash
uploaded_file = st.file_uploader("Choose a CSV file for checking review-rating discrepancy",type=["csv"])
```
* #### Performing the steps (4 to 6) for the uploaded test csv file
```bash
if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    df_test['Text'] = df_test['Text'].astype(str)
    df_test['Sentiment']=df_test['Text'].apply(sentiment_analyse)
    df_test['S_polarity'] = df_test['Text'].apply(find_pol)
    df_test_discrepency_reviews = df_test[(df_test['S_polarity'] > 0.65) & (df_test['Star'] <3) 
                                    & (df_test['Sentiment']=='Positive')]
```
* #### Display of reviews having good text reviews & negative ratings on web app using streamlit
```bash
st.write("The list of reviews where the reviews and ratings probably don't match are as below")
st.write(df_test_discrepency_reviews)
```


