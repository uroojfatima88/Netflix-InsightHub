import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# ------------------------
# Load data, model, scaler
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_data.csv")
    features = np.load("features.npy")
    return df, features

@st.cache_resource
def load_ann():
    return load_model("ann_model.h5")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.save")

df, features = load_data()
model = load_ann()
scaler = load_scaler()

# ------------------------
# TF-IDF vectorizer
# ------------------------
@st.cache_resource
def load_tfidf():
    tfidf = TfidfVectorizer(stop_words="english", max_features=500)
    tfidf.fit((df["listed_in"] + " " + df["description"]))
    return tfidf

tfidf = load_tfidf()

# ------------------------
# App Title
# ------------------------
st.title("Netflix Dataset Explorer 🎬")

# ------------------------
# Dataset Overview
# ------------------------
st.header("Dataset Overview")
num_movies = len(df[df['type'] == 'Movie'])
num_shows = len(df[df['type'] == 'TV Show'])
st.write(f"Total Movies: {num_movies}")
st.write(f"Total TV Shows: {num_shows}")

# ------------------------
# Explore Section
# ------------------------
st.header("Explore Netflix Titles")
search_term = st.text_input("Search for a title:")
if search_term:
    filtered_df = df[df["title"].str.contains(search_term, case=False, na=False)]
else:
    filtered_df = df

st.dataframe(filtered_df[["title", "type", "listed_in", "release_year", "description"]].head(50))

# ------------------------
# Recommendation Section
# ------------------------
st.header("Content Recommendations")
selected_title = st.selectbox("Select a title to get similar content", df["title"].values)

if st.button("Find Recommendations"):
    # Get the index of selected title
    idx = df[df["title"] == selected_title].index[0]

    # Get original TF-IDF vector and scale it
    content_text = df.loc[idx, "listed_in"] + " " + df.loc[idx, "description"]
    vector = tfidf.transform([content_text]).toarray()
    vector_scaled = scaler.transform(vector)

    # Compute cosine similarity with all scaled features
    sims = np.dot(features, vector_scaled.T).flatten() / (
        np.linalg.norm(features, axis=1) * np.linalg.norm(vector_scaled) + 1e-10
    )
    top_idx = sims.argsort()[-6:-1][::-1]  # top 5 similar titles
    recs = df.iloc[top_idx][["title", "type", "listed_in"]]

    st.write("### Recommended Titles:")
    for i, row in recs.iterrows():
        st.write(f"- **{row['title']}** ({row['type']}) - Genres: {row['listed_in']}")
