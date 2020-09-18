from flask import Flask, request, jsonify
from torch_utils import get_predication

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if (request.method == 'POST'):
        # Get the name
        name = request.args['name']
        # turn the name into tensor
        
        try:
            predication = get_predication(name)

            data = {'prediction': predication }

            # print(data)

            return jsonify(data)

        except Exception as e:
            # print(e)
            return jsonify({'error': 'Error during Predictions'})
        # predication 
        # return json

    return jsonify({'result': name})