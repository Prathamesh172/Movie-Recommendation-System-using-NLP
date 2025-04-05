## Movie Recommendation System

A content-based Movie Recommendation System built using NLP and machine learning techniques. Given a movie title, the system suggests 8 similar movies based on plot, genre, director, and lead actors.

---

### 🚀 Features

- Accepts movie titles as user input
- Recommends 8 similar movies using:
  - Movie Overview
  - Genre
  - Top 4 actors
- Text preprocessing with NLTK
- TF-IDF vectorization & cosine similarity for recommendations
- Deployed with Flask
- Ready for hosting on Render

---

### 🧠 Technologies Used

- Python 🐍
- Flask 🌐
- NLTK (Natural Language Toolkit)
- Pandas & NumPy
- Scikit-learn
- TF-IDF + Cosine Similarity
- HTML & CSS (for UI)

---

### 🌍 Live Demo

**Hosted on Render:** [Click here](https://your-render-link.com)  
_(Replace with actual link once deployed)_

---

### 📊 Dataset

- Dataset: `imdb_top_1000.csv`
- Source: Kaggle / IMDb Top 1000 Movies Dataset

Fields used:
- Series_Title, Overview, Genre, Director, Star1, Star2, Star3, Star4

---

### 💡 How It Works

1. **Input**: User enters a movie title
2. **Text Preprocessing**: Clean overview, genre, director, and actors
3. **Feature Engineering**: Combine all into one text string
4. **TF-IDF Vectorization**: Convert text into numeric vectors
5. **Cosine Similarity**: Find movies most similar to the input
6. **Output**: Top 8 recommendations displayed on the results page

---

### 🙌 Acknowledgements

- IMDb for the data
- NLTK for powerful NLP tools
- Scikit-learn for machine learning magic
- Flask for lightweight deployment
- Render for easy hosting

---


