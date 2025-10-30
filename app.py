# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
from pathlib import Path
import chardet
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# Estado global
model = None
scaler = None
encoders = {}
features = []
threshold = None
model_type = 'mlp'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_data', methods=['POST'])
def upload_data():
    global model, scaler, encoders, features, threshold, model_type
    
    if 'file' not in request.files:
        return jsonify(success=False, error="No se envió archivo")
    
    file = request.files['file']
    if file.filename == '':
        return jsonify(success=False, error="Archivo vacío")
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.csv')
    file.save(filepath)
    logger.info(f"Archivo guardado: {filepath}")

    try:
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext == '.xlsx':
            df = pd.read_excel(filepath, engine='openpyxl')
            logger.info("Archivo Excel leído correctamente")
        else:
            with open(filepath, 'rb') as f:
                raw = f.read(100000)
                encoding = chardet.detect(raw)['encoding'] or 'utf-8'
            df = pd.read_csv(
                filepath,
                delimiter=None,
                engine='python',
                on_bad_lines='skip',
                encoding=encoding,
                errors='replace'
            )
            logger.info(f"CSV leído con codificación: {encoding}")

        required = ['price', 'livingArea', 'lotSize', 'age', 'bedrooms', 'bathrooms', 'rooms', 'waterfront']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return jsonify(success=False, error=f"Faltan columnas: {', '.join(missing)}")

        threshold = df['price'].quantile(0.75)
        df['premium'] = (df['price'] > threshold).astype(int)

        cat_cols = ['waterfront', 'heating', 'newConstruction', 'centralAir']
        encoders.clear()
        for col in cat_cols:
            if col not in df.columns:
                df[col] = 'No'
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

        features = ['livingArea', 'lotSize', 'age', 'bedrooms', 'bathrooms', 'rooms', 
                    'waterfront', 'heating', 'newConstruction', 'centralAir']
        features = [f for f in features if f in df.columns]

        X = df[features]
        y = df['premium']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_type == 'rf':
            model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1500, random_state=42, early_stopping=True)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        return jsonify(
            success=True,
            message=f"Modelo {model_type.upper()} entrenado con {len(df)} propiedades",
            accuracy=f"{acc:.1%}",
            rows=len(df),
            premium_ratio=f"{y.mean()*100:.1f}%",
            report=report
        )

    except Exception as e:
        logger.error(f"Error en upload_data: {str(e)}")
        return jsonify(success=False, error=f"Error al procesar el archivo: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler, encoders, features, model_type, threshold
    if model is None:
        return jsonify(success=False, error="Entrena el modelo primero")

    data = request.json
    try:
        input_data = {
            'livingArea': float(data.get('livingArea', 1500)),
            'lotSize': float(data.get('lotSize', 0.5)),
            'age': float(data.get('age', 20)),
            'bedrooms': int(data.get('bedrooms', 3)),
            'bathrooms': float(data.get('bathrooms', 2)),
            'rooms': int(data.get('rooms', 7)),
            'waterfront': data.get('waterfront', 'No'),
            'heating': data.get('heating', 'hot air'),
            'newConstruction': data.get('newConstruction', 'No'),
            'centralAir': data.get('centralAir', 'No')
        }

        for col in encoders:
            val = input_data.get(col, 'No')
            if val in encoders[col].classes_:
                input_data[col] = encoders[col].transform([val])[0]
            else:
                input_data[col] = 0

        X = pd.DataFrame([input_data])[features]
        X_scaled = X if model_type == 'rf' else scaler.transform(X)
        
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0]
        confidence = max(prob) * 100

        recommendation = (
            "¡PROPIEDAD PREMIUM! Inversión de alto valor. Recomendada al 100%." 
            if pred else 
            "Propiedad estándar. Buena para vivir, pero no es premium."
        )

        return jsonify(
            success=True,
            isPremium=bool(pred),
            confidence=f"{confidence:.1f}%",
            recommendation=recommendation,
            estimatedPrice=int(threshold * (1.25 if pred else 0.75)),
            threshold=int(threshold)
        )
    except Exception as e:
        logger.error(f"Error en predict: {str(e)}")
        return jsonify(success=False, error=f"Error en predicción: {str(e)}")

@app.route('/train_model', methods=['POST'])
def train_model():
    global model_type
    model_type = request.json.get('type', 'mlp')
    return jsonify(success=True, message=f"Modelo cambiado a {model_type.upper()}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)