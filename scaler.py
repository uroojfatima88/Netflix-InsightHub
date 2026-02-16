import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load your cleaned dataset
df = pd.read_csv("cleaned_data.csv")
df.fillna("", inplace=True)

# Combine text features (same as ANN preprocessing)
df["content"] = df["listed_in"] + " " + df["description"]

# Convert text to vectors
tfidf = TfidfVectorizer(stop_words="english", max_features=500)
X = tfidf.fit_transform(df["content"]).toarray()

# Scale data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, "scaler.save")
print("Scaler file 'scaler.save' created successfully!")

# Optional: save the TF-IDF vectorizer too for Streamlit
joblib.dump(tfidf, "tfidf_vectorizer.save")
print("TF-IDF vectorizer 'tfidf_vectorizer.save' created successfully!")

