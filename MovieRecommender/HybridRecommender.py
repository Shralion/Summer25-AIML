import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
import surprise
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict

def createDataFrames():
    movies_df = pd.read_csv('ml-latest-small/movies.csv')
    ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
    tags_df = pd.read_csv('ml-latest-small/tags.csv')
    return movies_df, ratings_df, tags_df

def cleanMovieData(movies_df):
    movies_df['genres'] = movies_df['genres'].str.replace('|', ' ')
    movies_df['cleantitle'] = movies_df['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
    movies_df['release_year'] = movies_df['title'].str.extract(r'\((\d{4})\)', expand=False)
    movies_df['release_year'] = pd.to_numeric(movies_df['release_year'], errors = 'coerce')
    return movies_df

def cleanTagData(tags_df):
    tags_df['tag'] = tags_df['tag'].str.lower()
    tags_df.drop_duplicates(subset=['userId', 'movieId', 'tag'], inplace=True)
    return tags_df

def createMovieTagMatrix(movies_df, tags_df):
    tag_matrix = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    tag_matrix.columns = ['movieId', 'tags']
    # Merging the tags into the movies dataframe
    movies_df = movies_df.merge(tag_matrix, on='movieId', how='left')
    return movies_df

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
    for movie_index in missing_idx:
        sim_movies = cos_sim[movie_index]
        sim_movies[movie_index] = -1
        sim_movies[missing_idx.values] = -1      # exclusing movies with no tags
        # finding the top match
        most_sim_index = sim_movies.argmax()
        movies_df.at[movie_index, 'tags'] = movies_df.at[most_sim_index, 'tags']
    movies_df.drop(columns = ['combined'], inplace=True)
    return movies_df

def addAverageRatings(movies_df, ratings_df):
    avg_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.columns = ['movieId', 'avg_rating']
    movies_df = movies_df.merge(avg_ratings, on='movieId', how='left')
    movies_df['avg_rating'] = movies_df['avg_rating'].fillna(movies_df['avg_rating'].mean())  # Fill missing with overall mean
    return movies_df

def contentBasedFilter(movies_df):
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
    return movies_df, cos_sim

def createRatings(ratings_df):
    # Creating a Reader object to parse the ratings
    reader = Reader(rating_scale=(0.5, 5.0))
    ratings_data = Dataset.load_from_df(ratings_df, reader)
    return ratings_data

def collabFilterUser(movies_df, ratings_df, userID, model):
    all_movies = movies_df['movieId'].tolist()
    rated_movies = ratings_df[ratings_df['userId'] == userID]['movieId'].tolist()
    unrated_movies = []
    for movie in all_movies:
        if movie not in rated_movies:
            unrated_movies.append(movie)
    predictions = {}                            # dictionary movieId: predictRating
    for movie in unrated_movies:
        predictVal = model.predict(userID, movie)
        predictions[movie] = predictVal.est
    return predictions

def contentBasedUser(cos_sim, userID, ratings_df, movies_df):
    liked_movies = ratings_df[(ratings_df['userId'] == userID) & (ratings_df['rating'] >= 4.0)]
    liked_indices = liked_movies['movieId'].apply(lambda x: movies_df[movies_df['movieId'] == x].index[0]).tolist()
    scores = np.zeros(len(movies_df))
    for idx in liked_indices:
        scores += cos_sim[idx]
    if len(liked_indices) > 0:
        scores /= len(liked_indices)
    # normalizing to 0.5 - 5.0 scale
    min_score = scores.min()
    max_score = scores.max()
    normalized_scores = 0.5 + 4.5 * (scores - min_score) / (max_score - min_score + 1e-8)
    rated_movie_ids = set(liked_movies['movieId'])
    recommendations = {
        movies_df.iloc[i]['movieId']: normalized_scores[i]
        for i in range(len(scores))
        if movies_df.iloc[i]['movieId'] not in rated_movie_ids
    }
    return recommendations

def hybridRecommender(userID, ratings_df, movies_df, cos_sim, model, alpha=0.1):
    collab_scores = collabFilterUser(movies_df, ratings_df, userID, model)
    content_scores = contentBasedUser(cos_sim, userID, ratings_df, movies_df)
    hybrid_scores = {}
    all_movie_ids = set(collab_scores.keys()) | set(content_scores.keys())
    for movie_id in all_movie_ids:
        c_score = collab_scores.get(movie_id, 0)
        cb_score = content_scores.get(movie_id, 0)
        hybrid_scores[movie_id] = (alpha * c_score) + ((1 - alpha) * cb_score)
    # top 10
    top_10 = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    return top_10

movies_df, ratings_df, tags_df = createDataFrames()
movies_df = cleanMovieData(movies_df)
tags_df = cleanTagData(tags_df)
movies_df = createMovieTagMatrix(movies_df, tags_df)
movies_df = fill_tags(movies_df)
movies_df['tags'] = movies_df['tags'].fillna('')  # Just in case any tags are still NaN
movies_df = addAverageRatings(movies_df, ratings_df)

movies_df, cos_sim = contentBasedFilter(movies_df)

ratings_data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], Reader(rating_scale=(0.5, 5.0)))
trainset = ratings_data.build_full_trainset()
model = SVD(random_state = 42)
model.fit(trainset)

userID = 100
top_recommendations = hybridRecommender(userID, ratings_df, movies_df, cos_sim, model, alpha=0.7)

print(f"\nTop 10 movie recommendations for user {userID}:\n")
for movie_id, score in top_recommendations:
    title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
    print(f"{movie_id}: {title} (Score: {score:.2f})")

