from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Cargar el modelo entrenado y el escalador
model = joblib.load('modeloNeuR2.pkl')
scaler = joblib.load('DataScaled.pkl')
app.logger.debug('Modelo y transformadores cargados correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        year = float(request.form['year'])
        driven = float(request.form['km_driven'])
        engine = request.form['Engine (CC)'] 
        max_power = float(request.form['max_power (in bph)'])

        # Crear el DataFrame de entrada
        input_data = pd.DataFrame({
            'year': [year],
            'km_driven': [driven],
            'max_power (in bph)': [max_power],
            'Engine (CC)': [engine]
        })

        # Escalar los datos de entrada
        scaled_data = scaler.transform(input_data)

        # Realizar la predicción con los datos escalados
        prediction = model.predict(scaled_data)
        prediction_value = round(float(prediction[0]), 2)

        # Devolver la predicción como JSON
        return jsonify({'prediction': prediction_value})

    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
