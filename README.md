# 📰 Fake News Detection Using Machine Learning

## 📌 Overview

Fake News Detection is a Machine Learning project that classifies news articles as **Real** or **Fake** using Natural Language Processing (NLP) techniques. The project preprocesses news text, extracts features using TF-IDF Vectorization, and trains a Logistic Regression classifier to accurately identify fake news.

This project demonstrates an end-to-end machine learning pipeline, from data preprocessing and model training to prediction.

---

## 🚀 Features

- Data preprocessing and text cleaning
- Stopword removal and tokenization
- TF-IDF feature extraction
- Logistic Regression classifier
- Model evaluation using multiple metrics
- Predict whether a news article is **Real** or **Fake**
- Jupyter Notebook implementation

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Jupyter Notebook

---

## 📂 Project Structure

```
Fake-News-Detection/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── Fake_News_Detection.ipynb
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 📊 Workflow

1. Load the dataset
2. Clean and preprocess news articles
3. Convert text into numerical features using TF-IDF
4. Split the dataset into training and testing sets
5. Train a Logistic Regression model
6. Evaluate model performance
7. Predict whether new articles are Fake or Real

---

## 📈 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/hp30-trendz/Fake-News-Detection.git
```

Navigate to the project directory:

```bash
cd Fake-News-Detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Open the Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebook to:

- Load and preprocess the dataset
- Train the model
- Evaluate performance
- Make predictions

---

## 📚 Dataset

The project uses a labeled Fake News dataset containing:

- News Title
- News Content
- Label (Real/Fake)

Ensure the dataset is placed inside the `data/` folder before running the notebook.

---

## 📌 Future Improvements

- Deep Learning models (LSTM, GRU)
- Transformer-based models (BERT, RoBERTa)
- Real-time fake news detection
- Web application using Streamlit
- Explainable AI (SHAP/LIME)

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Harsh Patel**

GitHub: https://github.com/hp30-trendz

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
