from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np

app = Flask(__name__)

CORS(app)

print("Loading model and encoders...")

model = joblib.load('model/car_price_model.pkl')
label_encoders = joblib.load('model/label_encoders.pkl')

with open('model/model_config.json', 'r') as f:
    config = json.load(f)

print(f"Model loaded! R² Score: {config['r2_score']}")

# ── Route: Home Page ────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html', config=config)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_r2': config['r2_score'],
        'model_mae': config['mae']
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        brand = data.get('brand')
        location = data.get('location')
        year = int(data.get('year'))
        km_driven = float(data.get('kilometers_driven'))
        fuel_type = data.get('fuel_type')
        transmission = data.get('transmission')
        owner_type = data.get('owner_type')
        mileage = float(data.get('mileage'))
        engine = float(data.get('engine'))
        power = float(data.get('power'))
        seats = int(data.get('seats'))
        listed_price = float(data.get('listed_price'))

        car_age = config['current_year'] - year

        def safe_encode(encoder, value, col_name):
            try:
                return encoder.transform([value])[0]
            except ValueError:
                print(f"Warning: Unknown {col_name}: {value}")
                return -1

        brand_encoded = safe_encode(label_encoders['Brand'], brand, 'Brand')
        location_encoded = safe_encode(label_encoders['Location'], location, 'Location')
        fuel_type_encoded = safe_encode(label_encoders['Fuel_Type'], fuel_type, 'Fuel_Type')
        transmission_encoded = safe_encode(label_encoders['Transmission'], transmission, 'Transmission')
        owner_type_encoded = safe_encode(label_encoders['Owner_Type'], owner_type, 'Owner_Type')

        features = np.array([[
            car_age,
            km_driven,
            mileage,
            engine,
            power,
            seats,
            brand_encoded,
            location_encoded,
            fuel_type_encoded,
            transmission_encoded,
            owner_type_encoded
        ]])

        predicted_price = model.predict(features)[0]

        deviation = (listed_price - predicted_price) / predicted_price
        deviation_percent = round(deviation * 100, 2)

        threshold = config['threshold']

        if deviation > threshold:
            verdict = 'Overpriced'
        elif deviation < -threshold:
            verdict = 'Underpriced'
        else:
            verdict = 'Fair Price'

        abs_deviation = abs(deviation)
        if abs_deviation < 0.05:
            confidence = 'very high'
        elif abs_deviation < 0.10:
            confidence = 'high'
        elif abs_deviation < 0.20:
            confidence = 'medium'
        else:
            confidence = 'low'

        return jsonify({
            'success': True,
            'predicted_price': round(predicted_price, 2),
            'listed_price': listed_price,
            'verdict': verdict,
            'deviation_percent': deviation_percent,
            'confidence': confidence,
            'message': get_verdict_message(verdict, deviation_percent, predicted_price)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


def get_verdict_message(verdict, deviation_percent, predicted_price):
    if verdict == 'Overpriced':
        return f"This car is priced {abs(deviation_percent):.1f}% higher than its estimated fair value of Rs {predicted_price:.2f} Lakhs. Consider negotiating or looking at alternatives."
    elif verdict == 'Underpriced':
        return f"Great deal! This car is priced {abs(deviation_percent):.1f}% below its estimated fair value of Rs {predicted_price:.2f} Lakhs."
    else:
        return f"This car is fairly priced. The estimated fair value is Rs {predicted_price:.2f} Lakhs, which is within normal range."


# ── Route: Get Options ──────────────────────────────────────────────────────
@app.route('/options', methods=['GET'])
def get_options():
    return jsonify({
        'brands': config['brands'],
        'locations': config['locations'],
        'fuel_types': config['fuel_types'],
        'transmissions': config['transmissions'],
        'owner_types': config['owner_types']
    })


# ── Run the App ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("Starting Used Car Price Evaluator")
    print("=" * 50)
    print(f"Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop the server\n")

    app.run(debug=True, host='0.0.0.0', port=5000)