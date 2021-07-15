# Import necessary Libraries
from IPython.display import display
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# load the data and check the shape
df = pd.read_csv('D:\\Python_Projects\\Fake_News\\news.csv')  # NB. Change this directory to location of csv file
print(df.shape)
display(df.head())

# Get the Labels
labels = df['label']
print(labels.head())

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(df['text'],
                                                    labels,
                                                    test_size=0.2,
                                                    random_state=1313)

# initialize a TfidVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform the data
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Initialize a Passive Aggressive Classifier

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Predict on the test set and check accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy:{round(score*100, 2)}%')

# Create a confusion matrix to check TP and FN rate
display(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))
