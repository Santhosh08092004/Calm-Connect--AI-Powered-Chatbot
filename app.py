from flask import Flask, render_template, request, jsonify
import json
import random
import numpy as np
import re
import nltk
import pickle
import tensorflow as tf
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()

# Load saved assets
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
intents = json.load(open("intents.json", encoding="utf-8"))  # Ensure it's loaded with UTF-8 encoding
model = tf.keras.models.load_model("chatbot_model.keras")

# Clean and prepare input
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text.lower()

def chatbot_response(text):
    cleaned_text = clean_text(text)
    input_bag = [0] * len(words)
    input_words = nltk.word_tokenize(cleaned_text)
    input_words = [lemmatizer.lemmatize(word.lower()) for word in input_words]
    for word in words:
        if word in input_words:
            input_bag[words.index(word)] = 1

    prediction = model.predict(np.array([input_bag]))[0]
    tag = classes[np.argmax(prediction)]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "I'm not sure I understand..."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    response = chatbot_response(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
