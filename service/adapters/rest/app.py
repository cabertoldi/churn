import os
import sys
sys.path.append('./core/')

from flask import Flask, json, request
from index import is_churn

app = Flask(__name__)

@app.route('/predict-churn', methods=['POST'])
def predict_churn(): 
    customer = request.get_json()

    data = is_churn(customer)

    return app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)