# Recommender + Genre Classifier + Revenue Predictor for Movies/TV-Shows
## Structure:
``` 
├── Genre Classification/
├── MovieRecommender/
├── Popularity Predictor/
├── requirements.txt
└── README.md      <-- You are here
```
## Overview:
1. Revenue Predictor (for popularity prediction): 
   - Prediciting the revenue for a movie based on metadata such as budget, cast, director, writer, release date, etc.
   - Model used: XGBoost + TF-IDF
   - R^2 Score: 0.992
   - Dataset used: https://www.kaggle.com/datasets/danielgrijalvas/movies?resource=download&select=movies.csv
   - Sample output:
     ```
     Sample predictions vs. actual:
      Predicted: $54,296,944 | Actual: $54,641,191
      Predicted: $1,495,655 | Actual: $1,456,675
      Predicted: $42,757,144 | Actual: $42,593,455
      Predicted: $36,392,316 | Actual: $35,294,470
      Predicted: $3,705,168 | Actual: $3,560,932
     ```
       
2. Movie Recommender:
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
   - Bonus: A content-based recommender that computes similarity between movies based on metadata features such as genre, keywords, cast, and crew, to recommend similar content.
   - Sample output:
     ```
     Recommendations for movie ID 1: Toy Story (1995)
      78499: Toy Story 3 (2010)
      3114: Toy Story 2 (1999)
      3400: We're Back! A Dinosaur's Story (1993)
      2161: NeverEnding Story, The (1984)
      4886: Monsters, Inc. (2001)
     ```

3. Genre Classifier:
   - Multi-label genre classification using textual metadata such as cast, country, description, director, etc.
   - Pipeline: TF-IDF vectorization → MultiLabelBinarizer → One-vs-Rest Logistic Regression
   - Dataset Used: https://www.kaggle.com/datasets/shivamb/netflix-shows
   - Sample Output:
     ```
     Sample predictions with titles:
      Movie 1: Game Over, Man! → Predicted Genres: ['Comedies', 'Action & Adventure', 'Dramas']
      Movie 2: Arsenio Hall: Smart & Classy → Predicted Genres: ['Stand-Up Comedy', 'Documentaries', 'Dramas']
      Movie 3: Kazoops! → Predicted Genres: ["Kids' TV", 'International TV Shows', 'British TV Shows']
      Movie 4: We Are the Champions → Predicted Genres: ['Docuseries', 'International TV Shows', 'TV Dramas']
      Movie 5: Pablo Escobar, el patrón del mal → Predicted Genres: ['International TV Shows', 'Crime TV Shows', 'Spanish-Language TV Shows']
     ```

#### To Download the required libraries for all 3 projects:
   ```
   pip install -r requirements.txt
   ```
