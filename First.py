# Import necessary Libraries
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#load the data and check the shape
df=pd.read_csv('D:\\Python_Projects\\Fake_News\\news.csv')
print(df.shape)

# Get the Labels

labels = df['label']
print(labels.head())
