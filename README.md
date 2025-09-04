# 📊 Sentiment Analysis of E-commerce Product Reviews

## 📌 Project Overview
 This project uses Natural Language Processing (NLP) techniques to automatically classify e-commerce product reviews as Positive, Negative, or Neutral.
 It helps businesses monitor customer sentiment at scale and make data-driven decisions.

## 🎯 Objectives
- Analyze and visualize the distribution of customer sentiments.
- Clean and preprocess raw review text:
- Remove special characters
- Remove stopwords
- Apply stemming/lemmatization
- Convert text into numeric features using:
- Bag of Words (BoW)
- TF-IDF (Term Frequency–Inverse Document Frequency)
- N-grams
- Train and evaluate machine learning models:
- Random Forest
- Multinomial Naive Bayes
- Compare model performance using Macro F1-score and select the best model.

## 🗂️ Dataset
 The dataset contains product reviews collected from an e-commerce platform:
- Product ID → Unique identifier for each product
- Product Review → Customer feedback in text form
- Sentiment → Labeled sentiment (Positive, Negative, Neutral)

## ⚙️ Tech Stack
- Languages: Python
- Libraries: Pandas, NumPy, Scikit-learn, NLTK, spaCy, Matplotlib, Seaborn
- Models: Random Forest, Multinomial Naive Bayes

## 🔄 Workflow
 1. Exploratory Data Analysis (EDA)
- Analyze distribution of sentiments
 2. Text Preprocessing
- Clean text, remove stopwords, apply stemming/lemmatization
 3. Feature Engineering
- Vectorize text using BoW, TF-IDF, and N-grams
 4. Model Building
- Train/test split
- Train Random Forest & Naive Bayes models
 5. Evaluation
- Calculate Macro F1-score
- Generate classification report & confusion matrix

## 🚀 Future Improvements
 - Hyperparameter tuning for better performance
 - Experiment with alternative models: Logistic Regression, SVM, Deep Learning
 - Increase dataset size for more robust generalization
 - Use Transformer-based models (BERT, RoBERTa, GPT) for advanced sentiment analysis
