import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

netflix_df = pd.read_csv('netflix_titles.csv')

def cleanData(dataframe):
    dataframe.dropna(subset=['description', 'listed_in'], inplace=True)
    dataframe.drop(columns=['date_added', 'type', 'release_year'], inplace=True)
    dataframe['director'] = dataframe['director'].fillna('').str.lower()
    dataframe['cast'] = dataframe['cast'].fillna('').str.lower()
    dataframe['description'] = dataframe['description'].str.lower()
    # to know if it is kids/tv show
    dataframe['duration'] = dataframe['duration'].astype(str)
    # cleaning cast column
    dataframe['cast_clean'] = (dataframe['cast'].str.lower().str.split(',').apply(lambda names: [name.strip().replace(' ', '_') for name in names]).str.join(' '))
    dataframe['director_clean'] = (dataframe['director'].str.lower().str.split(',').apply(lambda names: [name.strip().replace(' ', '_') for name in names]).str.join(' '))
    dataframe['combined_cast'] = dataframe['director_clean'] + ' ' + dataframe['cast_clean']
    return dataframe

def fillCountry(dataframe):
    cast_tfidf = TfidfVectorizer(token_pattern=r"[^ ]+")
    cast_matrix = cast_tfidf.fit_transform(dataframe['combined_cast'])
    cos_sim = cosine_similarity(cast_matrix, cast_matrix)
    missing_country_index = dataframe[dataframe['country'].isna()].index
    for country_index in missing_country_index:
        sim_countries = cos_sim[country_index]
        sim_countries[country_index] = -1
        sim_countries[list(missing_country_index)] = -1
        most_sim_country = sim_countries.argmax()
        dataframe.at[country_index, 'country'] = dataframe.at[most_sim_country, 'country']
    return dataframe

def movie_types(dataframe):
    dataframe['is_TV'] = dataframe['duration'].str.contains("Season|Seasons", na=False).astype(int)
    dataframe['is_kids'] = dataframe['rating'].fillna("").str.contains("G|TV-Y").astype(int)
    dataframe['is_international'] = (~dataframe['country'].fillna("").str.lower().str.contains("united states")).astype(int)
    return dataframe

def createTargetGenres(dataframe):
    dataframe['listed_in'] = dataframe['listed_in'].apply(lambda x: [genre.strip() for genre in x.split(',')])
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(dataframe['listed_in'])
    return y, mlb

def vectorizeData(dataframe):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(dataframe['description'])
    X_meta = dataframe[['is_TV', 'is_kids', 'is_international']].values
    X_combined = hstack([X_tfidf, X_meta])
    return X_combined, tfidf

def trainClassifiers(X, y):
    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, indices, test_size=0.2, random_state=42
    )
    classifiers = []
    for i in range(y_train.shape[1]):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train[:, i])
        classifiers.append(clf)
    return classifiers, X_test, y_test, test_idx

def predictGenres(classifiers, X, mlb, topGenres=3):
    probs = np.array([clf.predict_proba(X)[:, 1] for clf in classifiers]).T
    top_preds = np.argsort(probs, axis=1)[:, -topGenres:]
    predictions = []
    for row in top_preds:
        labels = [mlb.classes_[i] for i in row[::-1]]
        predictions.append(labels)
    return predictions

def evaluateModel(classifiers, X_test, y_test):
    probs = np.array([clf.predict_proba(X_test)[:, 1] for clf in classifiers]).T
    y_pred = (probs >= 0.4).astype(int)
    print("F1-score:", f1_score(y_test, y_pred, average='micro'))

# Running code
netflix_df = cleanData(netflix_df)
netflix_df = fillCountry(netflix_df)
netflix_df = movie_types(netflix_df)
X, tfidf = vectorizeData(netflix_df)
y, mlb = createTargetGenres(netflix_df)
classifiers, X_test, y_test, test_idx = trainClassifiers(X, y)
evaluateModel(classifiers, X_test, y_test)
# Predict genres for test samples
predictions = predictGenres(classifiers, X_test[:10], mlb)
print("\nSample predictions with titles:")
for i, pred in enumerate(predictions):
    original_index = test_idx[i]
    title = netflix_df.iloc[original_index]['title']
    print(f"Movie {i + 1}: {title} → Predicted Genres: {pred}")
