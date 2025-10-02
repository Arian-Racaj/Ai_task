# SMS Spam Classifier

This is a small AI/ML project where I built a model to classify SMS messages as **spam** or **ham** (not spam).  
It also includes a simple **Flask API** so you can test your own messages.

# Dataset

I used the _UCI SMS Spam Collection_ dataset:

2 categories: `ham` and `spam`
Around 5,500 messages
Publicly available: [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

# Setup

1. Download or clone this project
2. Make sure Python is installed (tested on Python 3.10+)
3. Install required libraries:

pip install pandas nltk scikit-learn matplotlib flask
Download NLTK stopwords (first time only):

python
import nltk
nltk.download("stopwords")

How to Run

1.  First u need to activate the venv like this :
    source venv/Scripts/activate(after u finish the work u can deactivate)
    Run the main script (training & EDA)
    python text_classification.py

This will:

Load and clean the dataset

Show basic dataset info (total samples, category counts)

Save plots:

category_distribution.png (messages per category)

top_words.png (top 20 frequent words)

confusion_matrix_lr.png (Naive Bayes confusion matrix)

Train Naive Bayes and Logistic Regression

Print evaluation metrics for both models

2. Run Flask API

python text_classification.py
By default, the API runs on http://127.0.0.1:5000

curl -X POST http://127.0.0.1:5000/predict \
 -H "Content-Type: application/json" \
 -d '{"message":"Congratulations! You won a free prize!"}'

Example Response:

json
{
"prediction": "spam",
"confidence": 0.72
}
We can also send our own messages and see if they are classified as spam or ham so we can test it better with our own message.

# Testing the model behavior

Beyond the basic requirements, I conducted additional analysis to understand the model's behavior on challenging cases:

Tested short messages (under 30 characters) from the dataset
Analyzed how the model handles limited context
Discovered insights about model performance on edge cases
