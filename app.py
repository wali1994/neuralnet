from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Allows frontend from different domain (e.g., Render static site)

# Load model and tools
print("Loading model...")
model = tf.keras.models.load_model('mood_model.h5')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

# Mood info (emoji + recommendation) - from your original HTML
MOOD_INFO = {
    'amused': {
        'emoji': '😄',
        'recommendation': "You're feeling amused! Keep enjoying this light-hearted moment. Share laughter with others and engage in activities that bring you joy."
    },
    'angry': {
        'emoji': '😠',
        'recommendation': "You're feeling angry. Try taking deep breaths, going for a walk, or engaging in physical activity to channel this energy."
    },
    'disgusted': {
        'emoji': '🤢',
        'recommendation': "You're feeling disgusted. Take a step back from the situation. Engage in activities that refresh your mind and spirit."
    },
    'neutral': {
        'emoji': '😐',
        'recommendation': "You're feeling neutral. This is a balanced state perfect for reflection and planning. Use this time to assess your goals."
    },
    'sleepy': {
        'emoji': '😴',
        'recommendation': "You're feeling sleepy. Your body might need rest. Consider taking a nap or engaging in a relaxing activity."
    }
}

@app.route('/')
def home():
    return render_template('index.html')  # Optional: if you serve frontend from backend

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Preprocess and predict
    vec_text = vectorizer.transform([text]).toarray()
    prediction = model.predict(vec_text, verbose=0)[0]
    confidence = float(np.max(prediction) * 100)
    mood_idx = int(np.argmax(prediction))
    mood = le.inverse_transform([mood_idx])[0]

    info = MOOD_INFO.get(mood, MOOD_INFO['neutral'])

    return jsonify({
        'mood': mood,
        'confidence': round(confidence, 2),
        'emoji': info['emoji'],
        'recommendation': info['recommendation']
    })

if __name__ == '__main__':
    app.run(debug=True)