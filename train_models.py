import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# ---------- IVF Success Model ----------
ivf = pd.read_excel("ivf_datasets.xlsx", sheet_name="IVF_Success_Data")
X = ivf.drop("IVF_Success", axis=1)
y = ivf["IVF_Success"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the Random Forest Classifier with 100 decision trees
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
accuracy = accuracy_score(y_test, rf.predict(X_test))
print("IVF Model Accuracy:", accuracy)
joblib.dump(rf, "models/ivf_success_model.pkl")

# Save metadata
metadata = {
    "feature_order": list(X.columns),
    "model": "RandomForestClassifier",
    "accuracy": float(accuracy)
}
import json
with open("models/ivf_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

# ---------- Mood Trend Model ----------
mood = pd.read_excel("ivf_datasets.xlsx", sheet_name="Mood_Wellness_Data")
X = mood[["Mood_Rating", "Stress_Level", "Sleep_Hours", "Steps"]]
y = mood["Next_Day_Mood"]
lr = LinearRegression()
lr.fit(X, y)
joblib.dump(lr, "models/mood_trend_model.pkl")

# ---------- Emotion Analysis Model ----------
emo = pd.read_excel("ivf_datasets.xlsx", sheet_name="Emotion_Data")
cv = CountVectorizer()

# Split data for proper training and evaluation
X_text_train, X_text_test, y_label_train, y_label_test = train_test_split(
    emo["Text"], emo["Emotion"], test_size=0.2, random_state=42
)

X_train_vec = cv.fit_transform(X_text_train)
X_test_vec = cv.transform(X_text_test)

nb = MultinomialNB()
nb.fit(X_train_vec, y_label_train)
print("Emotion Model Accuracy:", accuracy_score(y_label_test, nb.predict(X_test_vec)))
joblib.dump(nb, "models/emotion_model.pkl")
joblib.dump(cv, "models/emotion_vectorizer.pkl")

print("✅ All models trained and saved successfully!")