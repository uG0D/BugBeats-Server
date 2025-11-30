import os

# CONFIGURACIÓN DE AMBIENTE PARA RENDER (Vital para evitar cuelgues)
os.environ['NUMBA_CACHE_DIR'] = '/tmp'
os.environ['MPLCONFIGDIR'] = '/tmp'
os.environ['NUMBA_NUM_THREADS'] = '1'

import numpy as np
import librosa
import io
import soundfile as sf
import requests
import tensorflow as tf
import gc
import time
from flask import Flask, request, jsonify

MODEL_FILE = 'modelo_ratas.tflite'
UBIDOTS_TOKEN = "BBUS-05HpL3CGv101KvETp3hGXsPHSGQuJ6"
DEVICE_LABEL = "bugbeats"
VARIABLE_LABEL = "rata"
MIN_CONFIDENCE = 0.80

app = Flask(__name__)
interpreter = None
input_details = None
output_details = None

print("--- 🧠 SERVIDOR INTELIGENTE (AUTO-RESAMPLE) ---")

try:
    if os.path.exists(MODEL_FILE):
        interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("✅ Modelo cargado.")
except Exception as e:
    print(f"❌ Error Modelo: {e}")

def enviar_a_ubidots(es_rata):
    try:
        val = 1.0 if es_rata else 0.0
        requests.post(
            f"https://stem.ubidots.com/api/v1.6/devices/{DEVICE_LABEL}",
            json={VARIABLE_LABEL: val},
            headers={"X-Auth-Token": UBIDOTS_TOKEN, "Content-Type": "application/json"},
            timeout=2
        )
    except: pass

@app.route('/', methods=['GET'])
def home():
    return "BugBeats Ready 🐀"

@app.route('/detectar', methods=['POST'])
def detectar():
    global interpreter
    if interpreter is None: return jsonify({"error": "Modelo off"}), 500

    print(f"\n📞 Recibido: {len(request.data)} bytes")
    
    try:
        # 1. Leer el audio original (sea cual sea su frecuencia)
        # sf.read devuelve los datos y la frecuencia original (samplerate)
        data, original_sr = sf.read(io.BytesIO(request.data))
        
        print(f"   -> Frecuencia original: {original_sr} Hz")

        # 2. RESAMPLING AUTOMÁTICO (La clave)
        # Si el Pico manda 10000Hz, lo subimos a 16000Hz para la IA
        if original_sr != 16000:
            print("   -> 🔄 Re-muestrando a 16000 Hz...")
            # Librosa.resample requiere float, aseguramos tipos
            if data.dtype != 'float32': data = data.astype('float32')
            # Transpuesta si es necesario para que librosa lo entienda
            if len(data.shape) > 1: data = data.mean(axis=1) # Mono
            
            # Resamplear
            data = librosa.resample(data, orig_sr=original_sr, target_sr=16000)
        
        # 3. Preparar para IA
        if data.dtype != 'float32': data = data.astype('float32')
        
        # Normalizar volumen
        max_val = np.max(np.abs(data))
        if max_val > 0: data = data / max_val

        # 4. Extraer características
        mfccs = librosa.feature.mfcc(y=data, sr=16000, n_mfcc=40)
        features = np.mean(mfccs.T, axis=0)
        
        input_data = np.array([features], dtype=np.float32)
        
        # 5. Predecir
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prob_rata = float(output_data[0][0])
        
        print(f"🧠 Resultado: {prob_rata*100:.1f}% Rata")

        es_rata = (prob_rata >= MIN_CONFIDENCE)
        enviar_a_ubidots(es_rata)
        
        # Limpieza
        del data
        del input_data
        gc.collect()

        return jsonify({
            "status": "ok", 
            "es_rata": 1 if es_rata else 0, 
            "mensaje": "RATA 🐀" if es_rata else "AMBIENTE 🍃",
            "confianza": prob_rata
        })

    except Exception as e:
        print(f"❌ Error Server: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
