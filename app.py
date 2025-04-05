import pandas as pd
import re
import nltk
from flask import Flask, request, render_template
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load dataset
df = pd.read_csv(r"C:\Users\Prathamesh\Downloads\Projects\Movie Recommendation System\movies rec\imdb_top_1000.csv")

# Combine actor columns
df['actors'] = df[['Star1', 'Star2', 'Star3', 'Star4']].apply(lambda x: ', '.join(x.dropna()), axis=1)
df.drop(columns=['Star1', 'Star2', 'Star3', 'Star4', 'Gross'], inplace=True)

# Handle missing values
df['Certificate'] = df['Certificate'].fillna(df['Certificate'].mode()[0])
df['Meta_score'] = df['Meta_score'].fillna(df['Meta_score'].mode()[0])

# Text preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text).lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

df['Overview_clean'] = df['Overview'].apply(preprocess_text)
df['Genre_clean'] = df['Genre'].apply(lambda x: ' '.join(x.replace(' ', '').split(',')))
df['Director_clean'] = df['Director'].apply(lambda x: x.replace(' ', ''))
df['Actors_clean'] = df['actors'].apply(lambda x: ' '.join([i.replace(' ', '') for i in x.split(', ')[:4]]))
df['combined_features'] = df['Overview_clean'] + ' ' + df['Genre_clean'] + ' ' + df['Director_clean'] + ' ' + df['Actors_clean']

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Case-insensitive recommendation function
def get_recommendations(title):
    title = title.lower().strip()
    df['Series_Title_Lower'] = df['Series_Title'].str.lower().str.strip()
    
    if title not in df['Series_Title_Lower'].values:
        return None
    
    idx = df[df['Series_Title_Lower'] == title].index[0]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:9]]
    return df.iloc[movie_indices]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        movie_title = request.form['title']
        try:
            recommendations = get_recommendations(movie_title)
            if recommendations is None:
                return render_template('index.html', error="Movie not found in database!")
            return render_template('results.html', movie=movie_title, recommendations=recommendations.to_dict('records'))
        except Exception as e:
            print("Error:", e)
            return render_template('index.html', error="An error occurred!")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
