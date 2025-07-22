# Recommender + Genre Classifier + Revenue Predictor for Movies/TV-Shows
## Structure:
``` 
├── GenreClassification/
├── MovieRecommender/
├── RevenuePrediction/
├── requirements.txt
└── README.md      <-- You are here
```
## Overview:
1. Revenue Predictor (for popularity prediction):
   - 
   - 
   - Dataset used:
   - Sample output:
     
3. Movie Recommender:
   - A hybrid recommender that uses a combination Collaborative filtering and Content-based filtering to recommend movies to users based on their preferences.
   - Models Used: TF-IDF + SVD
   - Dataset used: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
   - Sample output:
     ```
      Top 10 movie recommendations for user 100:
      898: Philadelphia Story, The (1940) (Score: 4.61)
      912: Casablanca (1942) (Score: 4.52)
      6711: Lost in Translation (2003) (Score: 4.42)
      88163: Crazy, Stupid, Love. (2011) (Score: 4.41)
     ```
