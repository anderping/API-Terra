from model_exe import TextPreprocessor, SentenceEmbedder, RequestClassifierPipeline, translate
from report import generate_report

import pandas as pd

import pickle
import os

from flask import Flask, jsonify, request, url_for
from flask_cors import CORS


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
app = Flask(__name__, static_folder=os.path.join(base_dir, 'static'))
app.config["DEBUG"] = True
CORS(app)

os.chdir(os.path.dirname(__file__))


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict the type of a request based on its text."""

    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl')

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    data = request.get_json()

    text = data.get('request_text', None)
    type = data.get('request_type', None)

    # Traduce a inglés si necesario
    en_text = translate(text) 

    sample = pd.DataFrame([{
        'request_text': en_text,
        'type': type
    }])

    prediction = model.predict(sample)

    return jsonify({'prediction': prediction['suggested_type'].item()}) # EN CASO DE QUE SIMPLEMENTE EL USUARIO HAYA PUESTO UN TYPE CONCRETO, SE LE DA A FRONTEND LA PREDICCIÓN
    # Y FULLSTACK SE ENCARGA DE LEVANTAR LA VENTANA QUE INDIQUE LA DISCREPANCIA Y MANEJARLA, DE FORMA QUE SI EL USUARIO QUIERE MANTENER LA SUYA SE PONGA UN TAG O ALGO
    # EN LA TARJETA DE CLICK-UP O SE ENVÍE UNA NOTIFICACIÓN DE ALGÚN TIPO AL PM. SI EL USUARIO LA ACEPTA SE CAMBIA EL TYPE POR EL PREDICHO.


@app.route('/report', methods=['POST'])
def report():
    """Endpoint to generate a report based on the provided data in JSON format."""
    # frequency = request.get_json('frequency', None) # FREQUENCY SE ESCOGE A TRAVÉS DEL FRONTEND, CON UN SELECTOR DE ALGÚN TIPO    
    data = request.get_json()
    
    try:
        json_graphs = generate_report(data)
        # urls = [url_for('static', filename=f'reports/{file_name}', _external=True) for file_name in filenames]
        
        return jsonify({"status": "ok", "graphs": json_graphs})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run()
