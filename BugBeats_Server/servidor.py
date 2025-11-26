import os
import joblib
import librosa
import numpy as np
import io
import requests
import wave
from flask import Flask, request, jsonify

# --- CONFIGURACIÓN ---
MODEL_FILE = 'modelo_ratas.pkl'
UBIDOTS_TOKEN = "BBUS-05HpL3CGv101KvETp3hGXsPHSGQuJ6"
DEVICE_LABEL = "bugbeats"
VARIABLE_LABEL = "rata"

app = Flask(__name__)

# Cargar Modelo
print("--- INICIANDO SERVIDOR BUG-BEATS ---")
try:
    if os.path.exists(MODEL_FILE):
        clf = joblib.load(MODEL_FILE)
        print(f"✅ Modelo cargado exitosamente.")
    else:
        print(f"❌ ERROR: No se encuentra '{MODEL_FILE}'.")
        clf = None
except Exception as e:
    print(f"❌ ERROR al cargar modelo: {e}")
    clf = None

def convertir_raw_a_wav(raw_bytes, sample_rate=16000):
    """
    Toma los bytes crudos del Pico y les agrega un encabezado WAV válido
    para que librosa pueda entenderlos.
    """
    try:
        wav_buffer = io.BytesIO()
        # Crear un archivo WAV en memoria
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)      # Mono
            wav_file.setsampwidth(2)      # 16 bits (2 bytes)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(raw_bytes)
        
        # Rebobinar el buffer al inicio para leerlo
        wav_buffer.seek(0)
        return wav_buffer
    except Exception as e:
        print(f"Error creando WAV header: {e}")
        return None

def procesar_audio(wav_file_obj):
    try:
        # Ahora librosa recibe un WAV válido
        data, sr = librosa.load(wav_file_obj, sr=16000, duration=6.0)
        mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
        features = np.mean(mfccs.T, axis=0)
        return features
    except Exception as e:
        print(f"❌ Error interno en librosa: {e}")
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
    return "BugBeats AI Server is Running & Ready for Raw Audio! 🐀"

@app.route('/detectar', methods=['POST'])
def detectar():
    if clf is None:
        return jsonify({"status": "error", "msg": "Modelo no cargado"}), 500

    print(f"\n📞 Recibida petición. Bytes crudos: {len(request.data)}")
    
    # 1. Convertir la "Carne Cruda" en "Hamburguesa" (WAV)
    wav_file = convertir_raw_a_wav(request.data)
    
    if wav_file is None:
        return jsonify({"status": "error", "msg": "Fallo al crear WAV header"}), 500

    # 2. Procesar
    features = procesar_audio(wav_file)
    
    if features is None:
        # Imprimimos el error en los logs de Render para que puedas verlo si falla
        return jsonify({"status": "error", "msg": "Error procesando audio con librosa"}), 500
    
    # 3. Predecir
    prediccion = clf.predict([features])[0]
    es_rata = int(prediccion) == 1
    
    enviar_a_ubidots(es_rata)
    
    return jsonify({
        "status": "ok", 
        "es_rata": int(prediccion),
        "mensaje": "RATA 🐀" if es_rata else "AMBIENTE 🍃"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
