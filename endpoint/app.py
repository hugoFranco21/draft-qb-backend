from flask import Flask
from flask import request, jsonify
from predictions import get_rookie_prediction, get_wins_prediction, get_should_draft
import json

app = Flask(__name__)

@app.route('/rookie-production', methods=['POST'])
def predict_rookies():
    data = request.data
    norm = get_rookie_prediction(data)
    output = {
        'success': False
    }
    if norm > 0:
        output = {
            'success': True,
            'prediction': norm
        }
    return json.dumps(output)

@app.route('/wins-expected', methods=['POST'])
def predict_wins():
    data = request.data
    norm = get_wins_prediction(data)
    output = {
        'success': False
    }
    if norm > 0:
        output = {
            'success': True,
            'prediction': norm
        }
    return json.dumps(output)

@app.route('/should-draft', methods=['POST'])
def should_draft():
    data = request.data
    norm = get_should_draft(data)
    output = {
        'success': False
    }
    if not norm == None:
        output = {
            'success': True,
            'prediction': norm
        }
    return json.dumps(output)

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)