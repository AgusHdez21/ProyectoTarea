from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import logging

app = Flask(__name__)

# Función para crear el modelo (si es necesaria)
def create_model_RFE():
    model = Sequential()
    model.add(Dense(64, input_dim=10, activation='relu'))  # Ajusta input_dim según tus datos
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Cargar el modelo entrenado y el escalador
try:
    model = joblib.load('modeloRf.pkl')
    scaler = joblib.load('DataScaled.pkl')
    app.logger.debug('Modelo y transformadores cargados correctamente.')
except Exception as e:
    app.logger.error(f'Error cargando el modelo o el escalador: {str(e)}')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        year = float(request.form['year'])
        driven = float(request.form['km_driven'])
        engine = float(request.form['engine'])
        max_power = float(request.form['max_power'])

        # Crear el DataFrame de entrada con todas las características necesarias
        input_data = pd.DataFrame({
            'Unnamed: 0': [0],               # Valor predeterminado
            'name': [0],                     # Valor predeterminado
            'year': [year],
            'km_driven': [driven],
            'fuel': [0],                     # Valor predeterminado
            'seller_type': [0],              # Valor predeterminado
            'owner': [0],                    # Valor predeterminado
            'seats': [0],                    # Valor predeterminado
            'max_power (in bph)': [max_power],
            'Mileage': [0],                  # Valor predeterminado
            'Engine (CC)': [engine],
            'Mileage Unit_km/kg': [0],       # Valor predeterminado
            'Mileage Unit_kmpl': [0],        # Valor predeterminado
            'transmission_Automatic': [0],   # Valor predeterminado
            'transmission_Manual': [0],      # Valor predeterminado
        })

        app.logger.debug(f'DataFrame de entrada creado:{input_data}')

        # Escalar los datos de entrada
        scaled_data = scaler.transform(input_data)

        scaled_data_for_prediction = scaled_data[:,[2,3,8,10]]
        
        # Realizar la predicción con los datos escalados
        prediction = model.predict(scaled_data_for_prediction)
        app.logger.debug(f'Prediccion:{prediction[0]}')

        #prediction_value = round(float(prediction[0]), 2)

        # Devolver la predicción como JSON
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
