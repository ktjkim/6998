import json
from flask import Flask, request, Response, jsonify, send_file
from inference import get_predictions
import torch
from constants import MODEL_PATH
from utils import download_model_from_gcs

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
download_model_from_gcs()
model = torch.load(MODEL_PATH)
model.eval()


@app.route('/health/', methods=['GET'])
def health_check():
    return Response(response=json.dumps({"status": 'healthy'}), status=200, mimetype="application/json")


@app.route('/predict/', methods=['POST'])
def main():
    request_json = request.get_json()
    request_instances = request_json['instances']
    sentence = request_instances['sentence']
    prediction = get_predictions(sentence=sentence, model=model)

    output = {'predictions':
        [
            {
                'result': prediction
            }
        ]
    }
    return jsonify(output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)