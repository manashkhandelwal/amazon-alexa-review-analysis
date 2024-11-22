# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# NLTK setup
import nltk
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
@st.cache
def load_data():
    return pd.read_csv('amazon_alexa.tsv', sep='\t')

df = load_data()

# Sidebar options
st.sidebar.title("Options")
analysis = st.sidebar.selectbox("Choose Analysis Type:", ["Overview", "EDA", "Sentiment Analysis", "Modeling"])

# Data Overview
if analysis == "Overview":
    st.title("Amazon Alexa Reviews Analysis")
    st.write("Dataset preview:")
    st.dataframe(df.head())
    st.write("Basic Information:")
    st.text(df.info())
    st.write("Dataset Shape:", df.shape)

# Exploratory Data Analysis
elif analysis == "EDA":
    st.title("Exploratory Data Analysis")
    
    # Rating Distribution
    st.subheader("Ratings Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='rating', data=df, ax=ax)
    st.pyplot(fig)

    # Variations Distribution
    st.subheader("Device Variations")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(y="variation", data=df, ax=ax, order=df['variation'].value_counts().index)
    st.pyplot(fig)

    # Feedback Distribution
    st.subheader("Feedback Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='feedback', data=df, ax=ax)
    st.pyplot(fig)

# Sentiment Analysis
elif analysis == "Sentiment Analysis":
    st.title("Sentiment Analysis")

    # Data Preprocessing
    def data_processing(text):
        text = text.lower()
        text = re.sub(r"http\S+www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s]', '', text)
        text_tokens = word_tokenize(text)
        filtered_text = [w for w in text_tokens if not w in stop_words]
        return " ".join(filtered_text)

    df['verified_reviews'] = df['verified_reviews'].apply(data_processing)

    # Display Word Clouds
    st.subheader("Word Clouds")

    # Positive Reviews
    pos_reviews = df[df.feedback == 1]
    text = ' '.join([word for word in pos_reviews['verified_reviews']])
    wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
    st.image(wordcloud.to_array(), caption="Positive Reviews Word Cloud")

    # Negative Reviews
    neg_reviews = df[df.feedback == 0]
    text = ' '.join([word for word in neg_reviews['verified_reviews']])
    wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
    st.image(wordcloud.to_array(), caption="Negative Reviews Word Cloud")

# Machine Learning Model
elif analysis == "Modeling":
    st.title("Modeling")

    # Feature Extraction
    cv = CountVectorizer()
    X = cv.fit_transform(df['verified_reviews'])
    Y = df['feedback']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    st.write("Training and testing split:")
    st.write(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    st.write(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    # Logistic Regression
    st.subheader("Logistic Regression")
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    logreg_pred = logreg.predict(x_test)
    logreg_acc = accuracy_score(logreg_pred, y_test)
    st.write(f"Test Accuracy: {logreg_acc * 100:.2f}%")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, logreg_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, logreg_pred))

    # Multinomial Naive Bayes
    st.subheader("Multinomial Naive Bayes")
    mnb = MultinomialNB()
    mnb.fit(x_train, y_train)
    mnb_pred = mnb.predict(x_test)
    mnb_acc = accuracy_score(mnb_pred, y_test)
    st.write(f"Test Accuracy: {mnb_acc * 100:.2f}%")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, mnb_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, mnb_pred))
