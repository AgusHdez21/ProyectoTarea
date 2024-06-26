from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import logging

app = Flask(__name__)

# Función para crear el modelo (si es necesaria, asegúrate de definirla correctamente)
def create_model_RFE():
    model = Sequential()
    model.add(Dense(64, input_dim=10, activation='relu'))  # Ajusta input_dim según tus datos
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

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
        engine = request.form['engine'] 
        max_power = float(request.form['max_power'])

        # Crear el DataFrame de entrada
        input_data = pd.DataFrame({
            'year': [year],
            'km_driven': [driven],
            'engine': [engine],
            'max_power': [max_power]
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
