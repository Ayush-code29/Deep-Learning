import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
     
df = pd.read_csv('train.txt',sep = ';',header = None,names = ['text','emotion'])
# print(df.head())
unique_emotions = df['emotion'].unique()
emotion_numbers = {}
i = 0
for emo in unique_emotions:
  emotion_numbers[emo] = i
  i +=1

df['emotion'] = df['emotion'].map(emotion_numbers)
df['text'] = df['text'].apply(lambda x : x.lower())
def remove_punc(txt):
  return txt.translate(str.maketrans('','',string.punctuation))
df['text'] = df['text'].apply(remove_punc)
def remove_numbers(txt):
    new = ""
    for i in txt:
        if not i.isdigit():
            new = new + i
    return new

df['text'] = df['text'].apply(remove_numbers)
def remove_emojis(txt):
    new = ""
    for i in txt:
        if i.isascii():
            new += i
    return new

df['text'] = df['text'].apply(remove_emojis)
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# print(df.loc[1]['text'])
def remove(txt):
  words = txt.split()
  cleaned = []
  for i in words:
    if not i in stop_words:
      cleaned.append(i)

  return ' '.join(cleaned)
     

df['text'] = df['text'].apply(remove)
# print(df.head())