import os
import numpy as np
import librosa
import io
import soundfile as sf
import requests
import tensorflow as tf
import gc # Importante para limpiar RAM
from flask import Flask, request, jsonify

# --- CONFIGURACIÓN ---
MODEL_FILE = 'modelo_ratas.tflite'
UBIDOTS_TOKEN = "BBUS-05HpL3CGv101KvETp3hGXsPHSGQuJ6"
DEVICE_LABEL = "bugbeats"
VARIABLE_LABEL = "rata"
MIN_CONFIDENCE = 0.80 

app = Flask(__name__)
interpreter = None
input_details = None
output_details = None

print("--- 🤖 SERVIDOR NEURONAL (TF CPU OPTIMIZADO) ---")

# Cargar modelo UNA sola vez al inicio
try:
    if os.path.exists(MODEL_FILE):
        interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"✅ Cerebro cargado.")
    else:
        print(f"❌ ERROR: Falta {MODEL_FILE}")
except Exception as e:
    print(f"❌ ERROR CARGA: {e}")

def procesar_audio(audio_data):
    # Normalización
    max_val = np.max(np.abs(audio_data))
    if max_val > 0: audio_data = audio_data / max_val

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0)
    return np.array([features], dtype=np.float32)

@app.route('/', methods=['GET'])
def home():
    return "BugBeats AI Alive"

@app.route('/detectar', methods=['POST'])
def detectar():
    global interpreter
    if interpreter is None: return jsonify({"error": "Modelo off"}), 500

    print(f"\n📞 Recibido: {len(request.data)} bytes")
    
    try:
        # 1. Leer WAV
        data, samplerate = sf.read(io.BytesIO(request.data))
        
        # Si es stereo, pasar a mono
        if len(data.shape) > 1: data = data.mean(axis=1)
        if data.dtype != 'float32': data = data.astype('float32')

        # 2. Inferencia
        input_data = procesar_audio(data)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prob_rata = float(output_data[0][0])
        
        print(f"🧠 Confianza: {prob_rata*100:.2f}% Rata")

        # 3. Ubidots (Con timeout corto para no bloquear)
        if prob_rata >= MIN_CONFIDENCE:
            try:
                requests.post(
                    f"https://stem.ubidots.com/api/v1.6/devices/{DEVICE_LABEL}",
                    json={VARIABLE_LABEL: 1.0},
                    headers={"X-Auth-Token": UBIDOTS_TOKEN, "Content-Type": "application/json"},
                    timeout=2
                )
            except: pass
        else:
            # Enviamos 0.0 a Ubidots también para confirmar que está vivo
             try:
                requests.post(
                    f"https://stem.ubidots.com/api/v1.6/devices/{DEVICE_LABEL}",
                    json={VARIABLE_LABEL: 0.0},
                    headers={"X-Auth-Token": UBIDOTS_TOKEN, "Content-Type": "application/json"},
                    timeout=2
                )
             except: pass

        # 4. LIMPIEZA DE MEMORIA OBLIGATORIA
        del data
        del input_data
        gc.collect() # Forzar al servidor a liberar RAM ya mismo

        return jsonify({
            "status": "ok", 
            "es_rata": 1 if prob_rata >= MIN_CONFIDENCE else 0,
            "mensaje": "RATA 🐀" if prob_rata >= MIN_CONFIDENCE else "AMBIENTE 🍃",
            "confianza": prob_rata
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
