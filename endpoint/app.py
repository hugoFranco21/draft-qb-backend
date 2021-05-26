from flask import Flask
from flask import request, jsonify
from predictions import get_rookie_prediction
import json

app = Flask(__name__)

@app.route('/rookie-production', methods=['POST'])
def predict_rookies():
    data = request.get_json()
    print(data)
    norm = get_rookie_prediction(data)
    output = {
        'success': False
    }
    if norm > 0:
        output = {
            'success': True,
            'prediction': norm
        }
    output = {
        'success': True
    }
    return json.dumps(output)

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)