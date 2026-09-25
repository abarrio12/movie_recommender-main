# 🎬 Building a Movie Recommendation System Using Flask and Machine Learning

Are you a movie buff who loves discovering new films? In this blog, I’ll walk you through how I built a **Movie Recommendation System** using **Flask**, **Python**, and **Machine Learning** — and even containerized it with **Docker**!

---

## 🚀 Project Overview

The goal of this project is to recommend similar movies based on a user's selection. It uses a content-based recommendation approach, meaning it suggests movies with similar content (genres, keywords, etc.) rather than relying on user ratings.

---
## 🤖 What is a Recommender System?

A **recommender system** is an algorithm that suggests items to users based on different factors like past behavior, preferences, or popularity.

### 🎯 Real-World Applications:

| Platform   | What it Recommends          |
|------------|------------------------------|
| **Netflix**   | Movies & TV Shows             |
| **Spotify**   | Songs & Podcasts              |
| **Amazon**    | Products                      |
| **YouTube**   | Videos                        |
| **LinkedIn**  | Jobs, People to connect with  |

---

## 🧭 Types of Recommender Systems

### 1. Content-Based Filtering
- Recommends items **similar** to what the user liked.
- Uses item features like genre, director, keywords.

### 2. Collaborative Filtering
- Recommends items liked by **similar users**.
- Doesn’t rely on content — only user behavior.
- ✅ **Your Task:** Implement **User-Based Collaborative Filtering** using techniques like:
  - Pearson correlation or cosine similarity between users
  - `surprise` library (SVD, KNNBasic)

### 3. Hybrid Methods
- Combines content-based and collaborative approaches.
- Often yields the most accurate and robust recommendations.
- ✅ **Your Task:** Try implementing a **hybrid approach** that:
  - Mixes user-based + content-based scores
  - Assigns weights or averages results from both models

---

## 📘 Project Breakdown

You’ll be working with two main files:

1. `analysis.ipynb`: For data loading, cleaning, feature extraction, modeling, evaluation.
2. `app.py`: For deploying your model with Flask.

---

## 🗂 Folder Structure

```
movie-recommender/
├── analysis.ipynb
├── app.py
├── requirements.txt
├── model.pkl
├── similarity.pkl
├── Dockerfile
├── templates/
    └── index.html
└── data/
    ├── tmdb_5000_credits
    └── tmdb_5000_movies
```

---

## 🛠 Tech Stack

- **Python**
- **Flask (Web Framework)**
- **Pandas & NumPy (Data Handling)**
- **Scikit-learn (Machine Learning)**
- **NLTK (Natural Language Processing)**
- **Docker (Containerization)**

---

## 🧪 TASKS – Predictive Modelling in `analysis.ipynb`

Use Jupyter Notebook (`analysis.ipynb`) to complete the following steps:

### ✅ Step 1: Load and Explore the Data
- Load `movies.csv`
- Clean and combine features for content-based modeling

### ✅ Step 2: Content-Based Filtering
- Create a `tags` column (genre, overview, cast, etc.)
- Vectorize using `CountVectorizer` or `TF-IDF`
- Compute **cosine similarity**

### ✅ Step 3: Save the Model

```python
import pickle
pickle.dump(movies, open('model.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
```

### ✅ Step 4: Test Recommendations
Create a function to recommend movies based on the selected title.

---
## BONUS TASKS – For Extra Learning
### ✅ Explore User-Based Collaborative Filtering

Use user-movie rating matrix (pick a new dataset of your choice for this task)  

Implement using pandas, or surprise library

### ✅ Build a Hybrid Recommender

Combine predictions from both methods

Use simple average or custom logic

---
## Move to Deployment – app.py
Once you're happy with the model in the notebook:

- Transfer working functions into app.py

- Use Flask to serve recommendations via a web interface

Use the existing structure in app.py:

- / route for homepage

- /recommend to show recommendations
---
## Make It Production-Ready with Docker  

After testing your app locally, Dockerise it using Dockerfile.  
---
## ▶ How to Run

### 🔧 Build Docker Image:
```bash
docker build -t movie-recommender .
```

### ▶ Run the Container:
```bash
docker run -p 5000:5000 movie-recommender
```

Then go to `http://localhost:5000` in your browser!

---

## 🎯 Future Improvements

- Add user-based collaborative filtering
- Deploy on a cloud platform like Heroku or AWS

---



## 📅 Submission Instructions

- ✅ **Deadline:** 20th April 2025  
- ✅ **What to Submit:** A link to your GitHub repository containing:
  - `analysis.ipynb`, `app.py`, `model.pkl`, `similarity.pkl`
  - Any additional data files or templates
  - *Optional:* `Dockerfile` and instructions in `README.md`

---

📝 **Please make sure your repository:**

- Has clear file structure  
- Is easy to run and test locally

---

📤 **Submit your GitHub link via the submission form on your course portal.**

