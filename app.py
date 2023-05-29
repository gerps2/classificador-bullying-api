from flask import Flask, request, jsonify
from business.predictor import Predictor
from langdetect import detect

app = Flask(__name__)
predictor = Predictor()

@app.route("/")
def check_api_online():
    with open('template/index.html', 'r') as file:
        content = file.read()
        
    return content

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    texts = data['texts']
    predictions = predictor.predict(texts)
    return jsonify(predictions)

@app.route('/api/detect-language', methods=['POST'])
def detect_language():
    data = request.get_json(force=True)
    text = data['text']
    language = detect(text)
    is_english = language == 'en'
    response = {'is_english': is_english}
    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
