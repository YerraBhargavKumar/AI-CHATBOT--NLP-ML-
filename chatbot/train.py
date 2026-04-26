
import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
with open("intents.json") as file:
    data = json.load(file)

texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = LogisticRegression()
model.fit(X, labels)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained!")