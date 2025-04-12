## 🧠 Sentiment Analysis Web App

A minimalist, dark-themed Flask web application that predicts **sentiment** (Positive, Neutral, or Negative) of user-submitted product reviews using a trained **Logistic Regression** model.

---

### 🚀 Features

- Accepts **product review** and **summary** as input
- Combines review and summary for better context
- Cleans & preprocesses text using regular expressions
- TF-IDF vectorization + Logistic Regression classifier
- Emoji-based output: 😊 😐 😠
- Simple, responsive UI
- Dockerized for deployment on **Hugging Face Spaces**

---

### 🧠 Technologies Used

- Python 🐍
- Flask 🌐
- Scikit-learn
- NLTK (for preprocessing)
- TF-IDF Vectorization
- Logistic Regression
- HTML/CSS (custom styling)

---

### 🌍 Demo

**Live on Hugging Face Spaces:** [Click here](https://huggingface.co/spaces/your-username/sentiment-app)

---

### 📁 Project Structure

```
sentiment-app/
│
├── app.py                  # Flask application
├── Dockerfile              # Docker setup
├── requirements.txt        # Python dependencies
├── sentiment_model.pkl     # Trained sentiment classifier
├── tfidf_vectorizer.pkl    # TF-IDF vectorizer
├── nltk_downloads.py       # (Optional) for Hugging Face NLTK compatibility
│
├── templates/
│   ├── index.html          # Landing page
│   └── predict.html        # Prediction form + result
│
└── README.md
```

---

### 🧠 Model Overview

The classifier is a **multi-class Logistic Regression model** trained on labeled product reviews.

Steps:
1. Clean review and summary text
2. Combine them for richer context
3. Vectorize using **TF-IDF**
4. Predict using **Logistic Regression**

Prediction Labels:
- `0` → Negative
- `1` → Neutral
- `2` → Positive

---

### 📦 Setup Instructions

#### 🔧 Local Run

1. Clone the repository
```bash
git clone https://github.com/your-username/sentiment-app.git
cd sentiment-app
```

2. Create virtual environment (optional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the app
```bash
python app.py
```

#### 🐳 Docker (for Hugging Face or local containers)

```bash
docker build -t sentiment-app .
docker run -p 7860:7860 sentiment-app
```

---

### 📊 Dataset

*Describe the dataset here — include source, number of records, preprocessing done, etc.*

> Example:
> The dataset used to train the model consists of ~25,000 labeled Amazon product reviews. Each review is labeled as Positive, Neutral, or Negative based on its overall tone. Text was cleaned using regex, lowercasing, and token filtering. Summary and main review body were concatenated before vectorization.

---

### ✨ Aesthetic Highlights

- Dark, modern interface
- Smooth hover transitions for buttons and cards
- Responsive layout
- Emoji-based result display for quick emotional grasp

---

### 📬 Contact

Built with ❤️ by Prathamesh

---

### 🧠 Future Improvements

- Add multilingual support (Hindi, Marathi, etc.)
- Support for voice input & audio sentiment
- Real-time dashboard for sentiment trends

---
