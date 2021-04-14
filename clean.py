
# %% import data and drop unnecessary columns
import pandas as pd
import re
import pandas as pd 
import numpy as np 

# if we do not add engine='python', it will throw out a Unicode error
df = pd.read_csv('training.1600000.processed.noemoticon.csv', header=None, sep=',', engine='python')

sentiments = df.iloc[:, 0]
text = df.iloc[:, 5]
sentiments = np.array(sentiments)
text = np.array(text)

data = {'sentiment': sentiments, 'text': text}

df = pd.DataFrame(data)
# %% Removing @User
def remove_pattern(text,pattern):
    r = re.findall(pattern,text)
    for i in r:
        text = re.sub(i,"",text)
    return text

df['text_cleaned'] = np.vectorize(remove_pattern)(df['text'], '@[\w]*')

df.head()

#%% Removing punctuation, numbers, and other unnecessary characters
df['text_cleaned'] = df['text_cleaned'].str.replace('[^a-zA-Z#]', ' ')

df.head(10)
# %% Removing short words (len() < 3) (is this necessary?)
df['text_cleaned'] = df['text_cleaned'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

df.head(10)

# %% Tokenization
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

tokenized_tweet = df['text_cleaned'].apply(lambda x: tknzr.tokenize(x))

tokenized_tweet.head(10)
# %% Stemming
from nltk import PorterStemmer

ps = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])

tokenized_tweet.head()

# %% Put the tokens in the array back to a string
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

df['text_cleaned'] = tokenized_tweet
df.head()

# %% Store the new data frame to a csv file

# change sentiment score to be 0 as negative and 1 as positive
df['sentiment'].replace(to_replace=4, value=1, inplace=True)

# shuffle the data (the original data is sorted by the sentiments, so that we cannot choose the first N rows to train)
df_shuffled=df.sample(frac=1).reset_index(drop=True)

df_shuffled.to_csv('data_cleaned.csv', index=False)
