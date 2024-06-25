from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

def create_model_RFE():
    model = Sequential()
    model.add(Dense(64, input_dim=10, activation='relu'))  # Ajusta input_dim según tus datos
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado y el escalador
model = joblib.load('modeloNeuR2.pkl')
scaler = joblib.load('dataSetScalado.pkl')

# Asegúrate de que la ruta del archivo encoder sea correcta
encoder = joblib.load('ordinal_encoder.pkl')
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
        fuel = request.form['fuel']  # Mantén como string para el encoder
        max_power = float(request.form['max_power'])

        # Verificar los datos recibidos
        app.logger.debug(f'year: {year}, km_driven: {driven}, fuel: {fuel}, max_power (in bph): {max_power}')

        # Crear el DataFrame de entrada
        input_data = pd.DataFrame({
            'Unnamed: 0': [0],
            'name': [0],
            'year': [year],
            'km_driven': [driven],
            'fuel': [fuel],
            'seller_type': [0],
            'owner': [0],
            'seats': [0],
            'max_power (in bph)': [max_power],
            'Mileage': [0], 
            'Engine (CC)': [0],
            'Mileage Unit_km/kg': [0],
            'Mileage Unit_kmpl': [0],
            'transmission_Automatic': [0],
            'transmission_Manual': [0],
        })

        # Convertir la columna 'fuel' usando el OrdinalEncoder
        input_data['fuel'] = encoder.transform(input_data[['fuel']])

        # Escalar los datos de entrada
        scaled_data = scaler.transform(input_data)

        # Seleccionar solo las características usadas para el modelo
        # Asegúrate de que los índices sean correctos según tu modelo
        scaled_data_for_prediction = scaled_data[:, [0, 1, 6, 7]]

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
