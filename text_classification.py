# text_classification.py
# Simple SMS Spam Classifier
# I made this project to detect if a text message is spam or not (ham).
# Includes a small Flask API so anyone can test messages.

import pandas as pd
import string
import nltk
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from flask import Flask, request, jsonify


# Here is the Setup & Load Dataset

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# Load dataset
df = pd.read_csv("SMSSpamCollection.csv", sep="\t", header=None, names=["label", "text"])

# Print some info about dataset
print(f"\nTotal messages: {len(df)}")
print("Messages per category:\n", df['label'].value_counts())


#  Preprocessing

def clean_text(text):
    # lowercase, remove punctuation, remove stopwords
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df["clean_text"] = df["text"].apply(clean_text)


#  Exploratory Analysis
# Category distribution chart
df["label"].value_counts().plot(kind="bar", title="Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("category_distribution.png")


# Here are top 20 most frequent words
all_words = " ".join(df["clean_text"]).split()
word_freq = Counter(all_words).most_common(20)
words, counts = zip(*word_freq)
plt.figure(figsize=(10,5))
plt.bar(words, counts)
plt.xticks(rotation=45)
plt.title("Top 20 Most Frequent Words")
plt.tight_layout()
plt.savefig("top_words.png")

# Convert text to numbers and split dataset

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train Models

# Naive Bayes (when i tested, this was the best one)
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

# Logistic Regression (i have used this one too for comparison)
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)


# Evaluate Models

print("\nNaive Bayes Performance:")
print(classification_report(y_test, nb_preds))

print("\nLogistic Regression Performance:")
print(classification_report(y_test, lr_preds))

print("\nAccuracy Comparison:")
print(f"Naive Bayes: {accuracy_score(y_test, nb_preds):.4f}")  # best one
print(f"Logistic Regression: {accuracy_score(y_test, lr_preds):.4f}")

# Confusion matrix for Naive Bayes
cm = confusion_matrix(y_test, nb_preds, labels=nb_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb_model.classes_)
disp.plot()
plt.title("Confusion Matrix - Naive Bayes")
plt.tight_layout()
plt.savefig("confusion_matrix_nb.png")



# Flask API

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # get message from user
    data = request.get_json()
    message = data.get("message", "")

    if not message.strip():
        return jsonify({"error": "Message is empty"}), 400

    # clean and transform message
    message_clean = clean_text(message)
    message_vect = vectorizer.transform([message_clean])

    # predict using Naive Bayes
    prediction = nb_model.predict(message_vect)[0]
    proba = nb_model.predict_proba(message_vect).max()

    return jsonify({"prediction": prediction, "confidence": float(proba)})

# Here i really wanted to do test some tricky examples from the dataset
# to see how well the model performs on them and what insights we can get

def analyze_tricky_dataset_examples():
    
    print("\n" + "="*60)
    print("Challenging Dataset Examples")
    print("="*60)
    
    # This will find the messages in the dataset that are SHORT and might be ambiguous
    df['length'] = df['clean_text'].str.len()
    short_messages = df[df['length'] < 30]  # We know that short messages can be tricky
    
    print(f"\nAnalyzing {len(short_messages)} short messages from dataset:")
    print("-" * 50)
    
    # Test how our model performs on these tricky short messages
    correct_predictions = 0
    tricky_examples = []
    
    for idx, row in short_messages.head(8).iterrows(): 
        original_text = row['text']
        true_label = row['label']
        
        # See what our model predicts
        pred = nb_model.predict(vectorizer.transform([row['clean_text']]))[0]
        proba = nb_model.predict_proba(vectorizer.transform([row['clean_text']]))
        confidence = proba.max()
        
    
        is_correct = pred == true_label
        status = "CORRECT" if is_correct else "WRONG"
        correct_predictions += 1 if is_correct else 0
        
        if pred != true_label or confidence < 0.7:  # this will show low-confidence or wrong predictions
            tricky_examples.append((original_text, true_label, pred, confidence))
            
        print(f"[{status}] '{original_text}'")
        print(f"   True: {true_label.upper()} | Pred: {pred.upper()} | Confidence: {confidence:.1%}")
    
    print(f"\nModel performance on short messages: {correct_predictions}/8 correct")
    
   
if __name__ == "__main__":
   
    analyze_tricky_dataset_examples()
    
    app.run(debug=True)