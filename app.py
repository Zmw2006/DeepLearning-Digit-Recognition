from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return 'Deep Learning Digit Recognition Demo'

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'prediction': '0', 'confidence': '0.99'})

if __name__ == '__main__':
    app.run(debug=True)
