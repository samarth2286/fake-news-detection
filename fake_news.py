# ── Section 1: Import Libraries ──────────────────────────────────
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ── Section 2: Load Dataset ──────────────────────────────────────
fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')

fake['label'] = 0   # 0 = Fake
real['label'] = 1   # 1 = Real

df = pd.concat([fake, real], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset shape:", df.shape)
print(df['label'].value_counts())

# ── Section 3: Preprocess Text ───────────────────────────────────
nltk.download('stopwords')
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text)).lower()
    tokens = [ps.stem(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

df['content'] = df['title'] + ' ' + df['text']
df['content'] = df['content'].apply(preprocess)
print("Preprocessing done!")

# ── Section 4: TF-IDF Vectorisation & Split ──────────────────────
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['content']).toarray()
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set size:", X_train.shape)
print("Testing set size: ", X_test.shape)

# ── Section 5: Train Logistic Regression ─────────────────────────
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# ── Section 6: Confusion Matrix ──────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Fake News Detection')
plt.tight_layout()
plt.show()

# ── Section 7: Save Model ────────────────────────────────────────
pickle.dump(model, open('fake_news_model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf_vec.pkl', 'wb'))
print("Model saved successfully!")

# ── Section 8: Predict on Custom Input ──────────────────────────
def predict_news(news_text):
    cleaned = preprocess(news_text)
    vector = tfidf.transform([cleaned]).toarray()
    prediction = model.predict(vector)[0]
    confidence = model.predict_proba(vector)[0]
    label = "REAL" if prediction == 1 else "FAKE"
    print(f"\nPrediction: {label}")
    print(f"Confidence → Fake: {confidence[0]*100:.2f}%  |  Real: {confidence[1]*100:.2f}%")

# Test with a sample
predict_news("Scientists discover new vaccine that cures all diseases overnight")
predict_news("The President signed a new infrastructure bill worth $1.2 trillion")