from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import joblib


print("Loading SMS Spam dataset...")
dataset = load_dataset("sms_spam")
split = dataset["train"].train_test_split(test_size=0.2, seed=42)  # same split as LSTM

X_train = split["train"]["sms"]
y_train = split["train"]["label"]
X_test  = split["test"]["sms"]
y_test  = split["test"]["label"]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# TF-IDF handles lowercasing, tokenization, and feature extraction
# class_prior=None lets the model learn from imbalanced data naturally,
# but TF-IDF's weighting already helps reduce majority-class dominance
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        stop_words="english",
        ngram_range=(1, 2),   # unigrams + bigrams (catches "free prize", "click here")
        max_features=10000,
    )),
    ("clf", MultinomialNB()),
])



pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham (Safe)", "Spam (Scam)"]))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot(cmap="Blues")
plt.title("Naive Bayes — Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_nb.png", dpi=150)
print("\nConfusion matrix saved to confusion_matrix_nb.png")


joblib.dump(pipeline, "scam-guard-naive-bayes.pkl")
print("Model saved to scam-guard-naive-bayes.pkl")