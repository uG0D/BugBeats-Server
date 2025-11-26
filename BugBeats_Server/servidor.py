import os
import joblib
import librosa
import numpy as np
import io
import requests
from flask import Flask, request, jsonify

# --- CONFIGURACIÓN ---
# En Render, a veces es difícil subir archivos grandes. 
# Si falla, asegúrate que el pkl esté en la misma carpeta.
MODEL_FILE = 'modelo_ratas.pkl'
UBIDOTS_TOKEN = "BBUS-YCozv61LfHSReGGecMnFLSWMr4STUe"
DEVICE_LABEL = "bugbeats"
VARIABLE_LABEL = "rata"

app = Flask(__name__)

# Cargar Modelo con manejo de errores robusto
print("--- INICIANDO SERVIDOR EN LA NUBE ---")
try:
    if os.path.exists(MODEL_FILE):
        clf = joblib.load(MODEL_FILE)
        print(f"✅ Modelo '{MODEL_FILE}' cargado exitosamente.")
    else:
        print(f"❌ ERROR: No se encuentra '{MODEL_FILE}' en {os.getcwd()}")
        clf = None # Evitar crash inmediato
except Exception as e:
    print(f"❌ ERROR al cargar modelo: {e}")
    clf = None

def procesar_audio_bytes(audio_data):
    try:
        audio_file = io.BytesIO(audio_data)
        data, sr = librosa.load(audio_file, sr=16000, duration=6.0)
        mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
        features = np.mean(mfccs.T, axis=0)
        return features
    except Exception as e:
        print(f"Error procesando audio: {e}")
        return None

def enviar_a_ubidots(es_rata):
    try:
        val = 1.0 if es_rata else 0.0
        url = f"https://stem.ubidots.com/api/v1.6/devices/{DEVICE_LABEL}"
        headers = {"X-Auth-Token": UBIDOTS_TOKEN, "Content-Type": "application/json"}
        requests.post(url, json={VARIABLE_LABEL: val}, headers=headers)
        print(f"✅ Ubidots actualizado: {val}")
    except Exception as e:
        print(f"⚠️ Error Ubidots: {e}")

@app.route('/', methods=['GET'])
def home():
    return "BugBeats AI Server is Running! 🐀"

@app.route('/detectar', methods=['POST'])
def detectar():
    if clf is None:
        return jsonify({"status": "error", "msg": "Modelo no cargado en servidor"}), 500

    print(f"\n📞 Recibida petición. Bytes: {len(request.data)}")
    features = procesar_audio_bytes(request.data)
    
    if features is None:
        return jsonify({"status": "error", "msg": "Audio corrupto"}), 400
    
    prediccion = clf.predict([features])[0]
    es_rata = int(prediccion) == 1
    
    enviar_a_ubidots(es_rata)
    
    return jsonify({
        "status": "ok", 
        "es_rata": int(prediccion),
        "mensaje": "RATA 🐀" if es_rata else "AMBIENTE 🍃"
    })

if __name__ == '__main__':
    # IMPORTANTE: En la nube, el puerto te lo da el entorno (os.environ)
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
