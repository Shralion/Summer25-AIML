import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.sparse import hstack
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, r2_score

movies_df = pd.read_csv('movies.csv')

def cleanData(dataframe):
    if 'released' in dataframe.columns:
        dataframe['release_year'] = pd.to_datetime(dataframe['released'], errors='coerce').dt.year
    if 'year' in dataframe.columns:
        dataframe['release_year'] = dataframe['release_year'].fillna(dataframe['year'])
    dataframe.drop(columns=['released'], inplace=True, errors='ignore')
    dataframe.dropna(subset=['gross', 'budget'], inplace=True)
    for col in ['director', 'writer', 'star', 'genre']:
        dataframe[col] = dataframe[col].fillna('').str.lower()
    for col in ['director', 'writer', 'star']:
        dataframe[col] = dataframe[col].str.split(',').apply(lambda names: [name.strip().replace(' ', '_') for name in names]).str.join(' ')
    dataframe['combinedCast'] = dataframe['director'] + ' ' + dataframe['writer'] + ' ' + dataframe['star']
    return dataframe

def fillVotesAndScores(df):
    vote_map = {}
    score_map = {}
    for role in ['director', 'writer', 'star']:
        all_people = df[role].str.split()
        for i, people in enumerate(all_people):
            for person in people:
                if person:
                    vote_map.setdefault(person, []).append(df.iloc[i]['votes'])         # checks if person is in map and appends the votes from movie to it
                    score_map.setdefault(person, []).append(df.iloc[i]['score'])        # same but for score, each person has a list of values mapped to them

    person_vote_avg = {k: np.nanmean(v) for k, v in vote_map.items()}                   # dictionary to store mean vote count for a person
    person_score_avg = {k: np.nanmean(v) for k, v in score_map.items()}                 # same but for score

    new_votes = []
    new_scores = []
    for _, row in df.iterrows():
        votes, scores = [], []
        for role in ['star', 'director', 'writer']:
            for person in row[role].split():
                if person in person_vote_avg:
                    votes.append(person_vote_avg[person])
                if person in person_score_avg:
                    scores.append(person_score_avg[person])
        avg_vote = np.nanmean(votes) if votes else np.nan
        avg_score = np.nanmean(scores) if scores else np.nan
        new_votes.append(avg_vote)
        new_scores.append(avg_score)

    df['votes'] = df['votes'].fillna(pd.Series(new_votes))
    df['score'] = df['score'].fillna(pd.Series(new_scores))
    df['votes'] = df['votes'].fillna(df['votes'].median())
    df['score'] = df['score'].fillna(df['score'].median())
    return df

def computeStarGross(df):
    person_gross = {}
    for role in ['star', 'director', 'writer']:
        all_people = df[role].str.split()
        flat_people = [person for sublist in all_people for person in sublist]
        unique_people = set(flat_people)
        role_gross = {}
        for person in unique_people:
            mask = df[role].str.contains(person)
            avg_gross = df[mask]['gross'].mean()
            role_gross[person] = avg_gross
        person_gross[role] = role_gross

    star_power_avg = []
    for _, row in df.iterrows():
        stars = row['star'].split()
        directors = row['director'].split()
        writers = row['writer'].split()
        star_gross = [person_gross['star'].get(s, 0) for s in stars]
        director_gross = [person_gross['director'].get(d, 0) for d in directors]
        writer_gross = [person_gross['writer'].get(w, 0) for w in writers]
        avg_star_gross = np.mean(star_gross) if star_gross else 0
        avg_director_gross = np.mean(director_gross) if director_gross else 0
        avg_writer_gross = np.mean(writer_gross) if writer_gross else 0
        star_power_avg.append((avg_star_gross, avg_director_gross, avg_writer_gross))
    df['star_gross'] = [x[0] for x in star_power_avg]
    df['director_gross'] = [x[1] for x in star_power_avg]
    df['writer_gross'] = [x[2] for x in star_power_avg]
    return df

def detectSequels(df):
    sequel_gross = []
    titles = df['name'].str.lower()
    for idx, row in df.iterrows():
        title = row['name'].lower()
        number = None
        for n in [' 2', ' 3', ' 4', ' 5', ' ii', ' iii', ' iv', ' part', ' episode', ' two', ' three', ' four']:
            if n in title:
                number = n
                break
        if number:
            base_title = title.split(number)[0].strip()
            possible_prequels = titles[titles.str.contains(base_title)]
            possible_prequels = possible_prequels[possible_prequels != title]
            if not possible_prequels.empty:
                prequel_idx = possible_prequels.index[0]
                prequel_gross = df.loc[prequel_idx, 'gross']
            else:
                prequel_gross = 0
        else:
            prequel_gross = 0
        sequel_gross.append(prequel_gross)
    df['prequel_gross'] = sequel_gross
    return df

def vectorizeCastGenre(dataframe):
    tfidf_cast = TfidfVectorizer(token_pattern=r"[^ ]+", max_features=500)
    tfidf_genre = TfidfVectorizer(token_pattern=r"[^ ]+")
    cast_matrix = tfidf_cast.fit_transform(dataframe['combinedCast'])
    genre_matrix = tfidf_genre.fit_transform(dataframe['genre'])
    return cast_matrix, genre_matrix


def predictRevenue(dataframe):
    cast_matrix, genre_matrix = vectorizeCastGenre(dataframe)
    dataframe['log_budget'] = np.log1p(dataframe['budget'])
    dataframe['log_votes'] = np.log1p(dataframe['votes'])
    dataframe['log_prequel_gross'] = np.log1p(dataframe['prequel_gross'])
    dataframe['budget_x_score'] = dataframe['budget'] * dataframe['score']
    dataframe['votes_x_score'] = dataframe['votes'] * dataframe['score']
    dataframe['gross_x_budget'] = dataframe['budget'] * dataframe['gross']

    features = [
        'log_budget', 'runtime',
        'star_gross', 'director_gross', 'writer_gross',
        'log_prequel_gross', 'score', 'log_votes', 'release_year',
        'budget_x_score', 'votes_x_score', 'gross_x_budget'
    ]
    num_features = dataframe[features].fillna(0).values
    num_scaled = StandardScaler().fit_transform(num_features)

    X = hstack([cast_matrix, genre_matrix, num_scaled])
    y = np.log1p(dataframe['gross'].values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
    }
    model = XGBRegressor(random_state=42, tree_method='hist')
    grid = GridSearchCV(model, param_grid, scoring=make_scorer(r2_score), cv=3)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    score = best_model.score(X_test, y_test)
    print(f"Log-Revenue R² (tuned): {score:.3f}")

    y_pred_log = best_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    print("\nSample predictions vs. actual:\n")
    for pred, actual in zip(y_pred[10:20], y_true[10:20]):
        print(f"Predicted: ${pred:,.0f} | Actual: ${actual:,.0f}")

    return best_model

movies_df = cleanData(movies_df)
movies_df = fillVotesAndScores(movies_df)
movies_df = computeStarGross(movies_df)
movies_df = detectSequels(movies_df)
model = predictRevenue(movies_df)