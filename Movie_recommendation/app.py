import pandas as pd
import ast
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("tmdb_5000_movies.csv")

# Preprocess data
df = df[['title', 'overview', 'genres']]
df['overview'] = df['overview'].fillna('')
df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)])
df['genres_str'] = df['genres'].apply(lambda x: ' '.join(x))
df['combined'] = df['overview'] + ' ' + df['genres_str']

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined'])
similarity = cosine_similarity(tfidf_matrix)

# Recommendation function
def recommend(movie_name):
    movie_name = movie_name.lower()
    if movie_name not in df['title'].str.lower().values:
        return []

    index = df[df['title'].str.lower() == movie_name].index[0]
    distances = similarity[index]
    movie_indices = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]
    return [df.iloc[i[0]]['title'] for i in movie_indices]

# Streamlit UI
st.title("🎬 Movie Recommendation System")
movie_input = st.text_input("Enter a movie name:")

if st.button("Recommend"):
    if movie_input:
        recommendations = recommend(movie_input)
        if recommendations:
            st.write(f"Top 5 movies similar to **{movie_input}**:")
            for movie in recommendations:
                st.write("👉", movie)
        else:
            st.error("❌ Movie not found in database!")
