# *** incorporating tags (filling meaningful tags) ***
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
from scipy.sparse import csr_matrix

movies_df = pd.read_csv('ml-latest-small/movies.csv')
ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
tags_df = pd.read_csv('ml-latest-small/tags.csv')

# Cleaning the data in the movies dataframe
movies_df['genres'] = movies_df['genres'].str.replace('|', ' ')
movies_df['cleantitle'] = movies_df['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()

# Cleaning the tags dataframe
tags_df['tag'] = tags_df['tag'].str.lower()
tags_df.drop_duplicates(subset=['userId', 'movieId', 'tag'], inplace=True)

# Getting the release date data from movie titles
movies_df['release_year'] = movies_df['title'].str.extract(r'\((\d{4})\)', expand=False)
movies_df['release_year'] = pd.to_numeric(movies_df['release_year'], errors = 'coerce')

# Creating a movie-tag mapping
tag_matrix = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
tag_matrix.columns = ['movieId', 'tags']

# Merging the tags into the movies dataframe
movies_df = movies_df.merge(tag_matrix, on='movieId', how='left')

# function to replace NaN values with meaningful tags
def fill_tags(movies_df):
    movies_df['combined'] = (
    movies_df['cleantitle'] + ' ' + 
    movies_df['genres'] + ' '
    )                                       # did not include release year

    tfidf = TfidfVectorizer(stop_words='english', max_df=0.8)
    movies_tfidf = tfidf.fit_transform(movies_df['combined'])
    cos_sim = cosine_similarity(movies_tfidf, movies_tfidf)

    missing_tags = movies_df[movies_df['tags'].isna()]
    missing_idx = movies_df[movies_df['tags'].isna()].index

    for movie_index in missing_tags.index:
        sim_movies = cos_sim[movie_index]
        sim_movies[movie_index] = -1
        sim_movies[list(missing_idx)] = -1      # exclusing movies with no tags
        # finding the top match
        most_sim_index = sim_movies.argmax()
        movies_df.at[movie_index, 'tags'] = movies_df.at[most_sim_index, 'tags']
    
    movies_df.drop(columns = ['combined'], inplace=True)

    return movies_df

# calling the function to fill the tags
movies_df = fill_tags(movies_df)

# filling empty tags (if any) with NaN as a safety net
movies_df['tags'] = movies_df['tags'].fillna('')

def addAverageRatings(movies_df, ratings_df):
    avg_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.columns = ['movieId', 'avg_rating']
    movies_df = movies_df.merge(avg_ratings, on='movieId', how='left')
    movies_df['avg_rating'] = movies_df['avg_rating'].fillna(movies_df['avg_rating'].mean())  # Fill missing with overall mean
    return movies_df

movies_df = addAverageRatings(movies_df, ratings_df)

title_tfidf = TfidfVectorizer(stop_words='english', max_df=0.8)
year_tfidf = TfidfVectorizer(stop_words='english', max_df=0.8)
tags_tfidf = TfidfVectorizer(stop_words='english', max_df=0.8)
genre_tfidf = TfidfVectorizer(stop_words='english', max_df=0.8)
# creating matrices for each component
title_matrix = title_tfidf.fit_transform(movies_df['cleantitle'])
year_matrix = year_tfidf.fit_transform(movies_df['release_year'].astype(str))
tags_matrix = tags_tfidf.fit_transform(movies_df['tags'])
genre_matrix = genre_tfidf.fit_transform(movies_df['genres'])
scaler = MinMaxScaler()
rating_feature = scaler.fit_transform(movies_df[['avg_rating']])
rating_matrix = csr_matrix(rating_feature)
# Adding weights
title_matrix = title_matrix.multiply(2.0)
genre_matrix = genre_matrix.multiply(3.0)
tags_matrix = tags_matrix.multiply(2.0)
year_matrix = year_matrix.multiply(1.0)
rating_matrix = rating_matrix.multiply(2.0)
# combining
combined = hstack([title_matrix, genre_matrix, year_matrix, tags_matrix, rating_matrix])
# Calculating the cosine similarity matrix
cos_sim = cosine_similarity(combined, combined)

# Function to get movie recommendations based on content similarity
def get_movie_recommendations(movie_id, num_recommendations=10):
    if movie_id not in movies_df['movieId'].values:
        raise ValueError(f"Movie ID {movie_id} not found in dataset.")
    idx = movies_df[movies_df['movieId'] == movie_id].index[0]
    similar_scores = list(enumerate(cos_sim[idx]))
    similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
    similar_scores = similar_scores[1:num_recommendations + 1]
    movie_indices = [i[0] for i in similar_scores]
    return movies_df.iloc[movie_indices][['movieId', 'title']]

# Testing the code with an example
movie_id = 1
recommendations = get_movie_recommendations(movie_id, num_recommendations=10)
title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
print(f"Recommendations for movie ID {movie_id}: {title}")
for idx, row in recommendations.iterrows():
    print(f"{row['movieId']}: {row['title']}")