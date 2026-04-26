from flask import Flask, render_template, request, jsonify
import pickle
import json
import random

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load intents data
with open("intents.json") as file:
    data = json.load(file)


# 🔥 MAIN CHATBOT FUNCTION (HYBRID)
def get_response(message):
    message = message.lower()

    # 🔹 Rule-based backup (handles common inputs)
    if any(word in message for word in ["hi", "hello", "hey"]):
        return random.choice([
            "Hello! 👋",
            "Hi there! 😊",
            "Hey! How can I help you?"
        ])

    if any(word in message for word in ["bye", "goodbye"]):
        return random.choice([
            "Goodbye! 👋",
            "See you later!",
            "Take care!"
        ])

    # 🔹 ML prediction
    X = vectorizer.transform([message])
    probs = model.predict_proba(X)[0]
    max_prob = max(probs)
    tag = model.classes_[probs.argmax()]

    # 🔹 Confidence check
    if max_prob < 0.3:
        return "Try saying hello, hi, or ask about AI 😊"

    # 🔹 Get response from intents
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "I didn't understand that."


# 🌐 Home route
@app.route("/")
def home():
    return render_template("index.html")


# 💬 Chat route
@app.route("/get", methods=["POST"])
def chatbot():
    user_text = request.form["msg"]
    response = get_response(user_text)
    return jsonify({"response": response})


# ▶️ Run app
if __name__ == "__main__":
    app.run(debug=True)