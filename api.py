from model_exe import TextPreprocessor, SentenceEmbedder, RequestClassifierPipeline, translate
import pandas as pd

import joblib
import os

from flask import Flask, jsonify, request, url_for
from flask_cors import CORS

import report


app = Flask(__name__)
app.config["DEBUG"] = True
CORS(app)

os.chdir(os.path.dirname(__file__))


@app.route('/predict', methods=['POST'])
def predict():
    with open('model.pkl', 'rb') as file:
        model = joblib.load(file) # ASEGURAR QUE ESE ES EL NOMBRE

    data = request.get_json()

    text = data.get('request_text', None) # ASEGURAR QUE ESE ES EL NOMBRE
    type = data.get('request_type', None)

    en_text = translate(text) # Traducir a inglés si necesario

    sample = pd.DataFrame([{
        'request_text': en_text,
        'type': type
    }])

    prediction = model.predict(sample)

    # if en_text is None:  # EL TEXTO TIENE QUE VENIR SIEMPRE COMPLETADO POR PARTE DE FRONTEND?
    #     return jsonify({'prediction': 'Not addressable'})

    # if type == "Other":
    #     return jsonify({'prediction': prediction['suggested_type'].item()})
         
    return jsonify({'prediction': prediction['suggested_type'].item()}) # EN CASO DE QUE SIMPLEMENTE EL USUARIO HAYA PUESTO UN TYPE CONCRETO, SE LE DA A FRONTEND LA PREDICCIÓN
    # Y FULLSTACK SE ENCARGA DE LEVANTAR LA VENTANA QUE INDIQUE LA DISCREPANCIA Y MANEJARLA, DE FORMA QUE SI EL USUARIO QUIERE MANTENER LA SUYA SE PONGA UN TAG O ALGO
    # EN LA TARJETA DE CLICK-UP O SE ENVÍE UNA NOTIFICACIÓN DE ALGÚN TIPO AL PM. SI EL USUARIO LA ACEPTA SE CAMBIA EL TYPE POR EL PREDICHO.


@app.route('/report', methods=['POST'])
def report():
    frequency = request.get_json('frequency', None) # FREQUENCY SE ESCOGE A TRAVÉS DEL FRONTEND, CON UN SELECTOR DE ALGÚN TIPO

    try:
        filenames = report.generate_report(frequency)
        urls = [url_for('static', filename=f'reports/{file_name}', _external=True) for file_name in filenames]
        
        return jsonify({"status": "ok", "graphs": urls})  # COMENTAR CON FRONTEND QUE LO QUE LES MANDO SON URLS
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run()
