import nltk
import joblib
import re
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download("stopwords")
model = joblib.load('sentiment_model.pkl')
cv=joblib.load('count_vectorizer.pkl')
ps=PorterStemmer()
def preprocess(text):
    # Clean the review text
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    all_stopwords = stopwords.words('english')
    if 'not' in all_stopwords:
        all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review

st.title("Restaurant Review Sentiment Analysis")
st.write("Enter a restaurant review to predict its sentiment (positive or negative).")
review=st.text_area("Enter your review here:")
if st.button("Predict"):
    cleaned_review = preprocess(review)

    vector = cv.transform([cleaned_review]).toarray()

    predict = model.predict(vector)

    if predict == 1:
        st.success("The review is positive.")
    else:
        st.error("The review is negative.")