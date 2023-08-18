import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

tweets = pd.read_csv("tweets.csv")
stopwords = pd.read_csv("stopwords.txt", header=None)[0].tolist()


def preprocess_text(text):
    # Remove non-alphanumeric characters (except spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Lowercase
    text = text.lower()
    # Split into tokens
    tokens = text.split()
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords]
    # Rejoin tokens
    text = " ".join(tokens)
    return text


# Preprocess the train set
tweets["text"] = tweets["text"].apply(preprocess_text)

# Create a TF-IDF vectorizer
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(tweets["text"])

# Split the train set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, tweets["target"], test_size=0.2, random_state=42
)

# Train a Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Evaluate the model using the validation set
score = lr.score(X_val, y_val)
print(f"Validation set accuracy: {score}")

# Predict probabilities using the Logistic Regression model
y_pred_lr = lr.predict_proba(X_val)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_val, y_pred_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Generate random predictions as a baseline
y_pred_random = np.random.rand(len(y_val))
fpr_random, tpr_random, _ = roc_curve(y_val, y_pred_random)
roc_auc_random = auc(fpr_random, tpr_random)

# Plot the ROC curves for Logistic Regression and Random
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {roc_auc_lr:.2f})")
plt.plot(fpr_random, tpr_random, label=f"Random (AUC = {roc_auc_random:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
