from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('modeloNeuronaR2.pkl')
scaler = joblib.load('dataFrameScalado.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        Schooling = float(request.form['Schooling'])
        HIVAIDS = float(request.form['HIVAIDS'])
        healthexppercapita = float(request.form['healthexppercapita'])
        fertilityrate = float(request.form['fertilityrate'])

        # Verificar los datos recibidos
        app.logger.debug(f'Schooling: {Schooling}, HIVAIDS: {HIVAIDS}, healthexppercapita: {healthexppercapita}, fertilityrate: {fertilityrate}')

        input_data = pd.DataFrame({
            'Schooling': [Schooling],
            'HIV.AIDS': [HIVAIDS],
            'Status': [0],
            'wateraccess': [0],
            'tuberculosis': [0],
            'inflation': [0],
            'healthexppercapita': [healthexppercapita],
            'fertilityrate': [fertilityrate],
            'CO2': [0],
            'urbanpopgrowth': [0], 
            'leastdeveloped': [0]
        })

        # Escalar los datos de entrada
        scaled_data = scaler.transform(input_data)

        # Seleccionar solo las características usadas para el modelo
        scaled_data_for_prediction = scaled_data[:, [0, 1, 6, 7]]  # Asegúrate de que estos índices son correctos

        # Realizar la predicción con los datos escalados
        prediccion = model.predict(scaled_data_for_prediction)

        # Devolver la predicción como JSON
        prediction_value = round(float(prediccion[0]), 2)

        return jsonify({'prediction': prediction_value})

    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
