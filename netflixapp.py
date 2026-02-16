import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# ------------------------
# File paths
# ------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "cleaned_data.csv")
FEATURES_PATH = os.path.join(BASE_DIR, "features.npy")
MODEL_PATH = os.path.join(BASE_DIR, "ann_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.save")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.save")

# ------------------------
# Load data and resources
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    features = np.load(FEATURES_PATH)
    return df, features

@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    return model, scaler, tfidf

df, features = load_data()
model, scaler, tfidf = load_resources()

# ------------------------
# App Title
# ------------------------
st.title("Netflix Dataset Explorer & Recommendations 🎬")

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
    # Get text of selected title
    idx = df[df["title"] == selected_title].index[0]
    content_text = df.loc[idx, "listed_in"] + " " + df.loc[idx, "description"]

    # Convert text to TF-IDF and scale
    vector = tfidf.transform([content_text]).toarray()
    vector_scaled = scaler.transform(vector)

    # Compute cosine similarity with all embeddings
    sims = np.dot(features, vector_scaled.T).flatten() / (
        np.linalg.norm(features, axis=1) * np.linalg.norm(vector_scaled) + 1e-10
    )

    # Top 5 similar titles
    top_idx = sims.argsort()[-6:-1][::-1]
    recs = df.iloc[top_idx][["title", "type", "listed_in"]]

    st.write("### Recommended Titles:")
    for i, row in recs.iterrows():
        st.write(f"- **{row['title']}** ({row['type']}) - Genres: {row['listed_in']}")
