import pandas as pd
import numpy as np
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Text preprocessing function (same as in app.py)
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load and preprocess data
print("Loading data...")
df = pd.read_csv('amazon_reviews.csv')

# Rename columns
df.rename(columns={'label':'target','Reviews':'text'}, inplace=True)

# Encode labels
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Apply text preprocessing
print("Preprocessing text...")
df['transformed_text'] = df['text'].apply(transform_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['transformed_text'], df['target'], test_size=0.2, random_state=2)

# Create TF-IDF vectorizer
print("Creating TF-IDF vectorizer...")
tfidf = TfidfVectorizer(max_features=3000)

# Fit and transform training data
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train Logistic Regression model
print("Training Logistic Regression model...")
lr = LogisticRegression(solver='liblinear', penalty='l1')
lr.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = lr.predict(X_test_tfidf)
from sklearn.metrics import accuracy_score, precision_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f"Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")

# Save model and vectorizer
print("Saving model and vectorizer...")
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(lr, open('model.pkl', 'wb'))

print("Model training completed! model.pkl and vectorizer.pkl have been saved.") 